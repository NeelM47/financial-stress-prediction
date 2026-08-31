import os, random, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(42)

print("Loading data...")
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
y = train[TARGET].values

cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern", "demo_fingerprint"]

def engineer_features(df, is_train=True):
    df = df.copy()
    # 1. Demographic Fingerprint
    df['demo_fingerprint'] = df['region'].astype(str) + "_" + df['segment'].astype(str) + "_" + df['earning_pattern'].astype(str)

    # 2. ID Leak Features
    if ID_COL in df.columns:
        df['id_length'] = df[ID_COL].apply(len)
        df['id_numeric'] = df[ID_COL].apply(lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else 0)
        df = df.drop(columns=[ID_COL])
        
    if is_train and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    # 3. Customer Profile Hash for GroupKFold
    if is_train:
        df["profile_hash"] = df["age"].astype(str) + "_" + df["gender"].astype(str) + "_" + df["region"].astype(str)

    # 4. Extract Temporal Matrix
    temporal_metrics = {}
    balance_cols = []
    for col in df.columns:
        if re.match(r"^m[1-6]_daily_avg_bal$", col):
            balance_cols.append(col)
        m = re.match(r"^m([1-6])_([a-z_]+)_(volume|total_value|highest_amount)$", col)
        if m:
            temporal_metrics.setdefault((m.group(2), m.group(3)), {})[int(m.group(1))] = col
        m2 = re.match(r"^m([1-6])_([a-z_]+)_([a-z_]+)$", col)
        if m2 and m2.group(3) in ("companies", "merchants", "banks", "recipients", "senders", "agents"):
            temporal_metrics.setdefault((m2.group(2), m2.group(3)), {})[int(m2.group(1))] = col

    new_features = {}
    
    # 5. Temporal Aggregations (Slopes, EMA, Deltas)
    for key, month_map in temporal_metrics.items():
        vals = np.column_stack([df[month_map.get(m)].values if month_map.get(m) else np.zeros(len(df)) for m in range(1, 7)])
        months = np.arange(1, 7).astype(float)
        
        prefix = f"{key[0]}_{key[1]}"
        new_features[f"{prefix}_ema"] = np.average(vals, axis=1, weights=np.array([6, 5, 4, 3, 2, 1]))
        new_features[f"{prefix}_slope"] = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in vals])
        new_features[f"{prefix}_mean"] = vals.mean(axis=1)
        new_features[f"{prefix}_std"] = vals.std(axis=1)
        new_features[f"{prefix}_recent_old_ratio"] = np.divide(vals[:, 0], vals[:, 5], out=np.zeros_like(vals[:, 0], dtype=float), where=vals[:, 5] != 0)
        new_features[f"{prefix}_dormant_months"] = (vals == 0).sum(axis=1)

        for m in range(5):
            new_features[f"{prefix}_delta_m{m+1}"] = vals[:, m] - vals[:, m + 1]

    # 6. Balance Specific Aggregations
    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        new_features["bal_slope"] = np.array([np.polyfit(np.arange(1, 7).astype(float), v, 1)[0] if np.any(v != 0) else 0.0 for v in bal_vals])
        new_features["bal_mean"] = bal_vals.mean(axis=1)
        new_features["bal_volatility"] = bal_vals.std(axis=1)
        new_features["bal_min"] = bal_vals.min(axis=1)
        new_features["bal_decline"] = (bal_vals[:, 0] < bal_vals[:, 5]).astype(float)

    # 7. Total Volumes and Values
    txn_types = ["paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"]
    for m in range(1, 7):
        vol_list = [df[f"m{m}_{t}_volume"].values for t in txn_types if f"m{m}_{t}_volume" in df.columns]
        if vol_list: new_features[f"m{m}_total_volume"] = np.column_stack(vol_list).sum(axis=1)
        
        val_list = [df[f"m{m}_{t}_total_value"].values for t in txn_types if f"m{m}_{t}_total_value" in df.columns]
        if val_list: new_features[f"m{m}_total_value"] = np.column_stack(val_list).sum(axis=1)
        
        if f"m{m}_withdraw_total_value" in df.columns and f"m{m}_deposit_total_value" in df.columns:
            new_features[f"m{m}_wd_ratio"] = np.divide(df[f"m{m}_withdraw_total_value"].values, df[f"m{m}_deposit_total_value"].values, out=np.zeros(len(df)), where=df[f"m{m}_deposit_total_value"].values != 0)

    # 8. ARPU & MACD Features
    new_features['arpu_to_m1_bal'] = df['arpu'] / (df['m1_daily_avg_bal'] + 1e-5)
    new_features['arpu_to_m1_withdraw'] = df['arpu'] / (df['m1_withdraw_total_value'] + 1e-5)
    new_features['x_90_d_activity_to_arpu'] = df['x_90_d_activity_rate'] / (df['arpu'] + 1e-5)

    if balance_cols:
        new_features['bal_macd_ratio'] = bal_vals[:, 0:2].mean(axis=1) / (bal_vals[:, 3:6].mean(axis=1) + 1e-5)
        
    for prefix in ['deposit_total_value', 'withdraw_total_value', 'received_total_value']:
        m1, m2, m4, m5, m6 = f'm1_{prefix}', f'm2_{prefix}', f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}'
        if all(c in df.columns for c in [m1, m2, m4, m5, m6]):
            new_features[f'{prefix}_macd_ratio'] = df[[m1, m2]].mean(axis=1) / (df[[m4, m5, m6]].mean(axis=1) + 1e-5)

    # 9. True Net Cashflow
    inflow_types = ['deposit', 'received', 'transfer_from_bank']
    outflow_types = ['withdraw', 'merchantpay', 'paybill', 'mm_send']

    for m in range(1, 7):
        inflow = sum([df[f'm{m}_{t}_total_value'].values for t in inflow_types if f'm{m}_{t}_total_value' in df.columns])
        outflow = sum([df[f'm{m}_{t}_total_value'].values for t in outflow_types if f'm{m}_{t}_total_value' in df.columns])
        
        if isinstance(inflow, np.ndarray):
            new_features[f'm{m}_net_cashflow'] = inflow - outflow
            new_features[f'm{m}_cashflow_margin'] = (inflow - outflow) / (inflow + 1e-5)
            new_features[f'm{m}_cashflow_to_arpu'] = (inflow - outflow) / (df['arpu'] + 1e-5)

    if 'm1_daily_avg_bal' in df.columns:
        new_features['is_m1_bankrupt'] = (df['m1_daily_avg_bal'] == 0).astype(float)
        new_features['is_m2_bankrupt'] = (df['m2_daily_avg_bal'] == 0).astype(float)

    cashflow_matrix = np.column_stack([new_features[f'm{m}_net_cashflow'] for m in range(1, 7)])
    new_features['net_cashflow_slope'] = np.array([np.polyfit(np.arange(1, 7).astype(float), v, 1)[0] if np.any(v != 0) else 0.0 for v in cashflow_matrix])

    # 10. Compile and Apply 90-Day Alignment
    df_new = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, df_new], axis=1)

    for prefix in ['daily_avg_bal', 'net_cashflow', 'deposit_total_value', 'withdraw_total_value']:
        if f'm1_{prefix}' in result.columns:
            recent_90d = result[[f'm1_{prefix}', f'm2_{prefix}', f'm3_{prefix}']].mean(axis=1)
            prior_90d = result[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1)
            result[f'{prefix}_90d_trend'] = recent_90d / (prior_90d + 1e-5)
            result[f'{prefix}_90d_delta'] = recent_90d - prior_90d
            
    for m in range(1, 4):
        paybill_col, merchant_col, withdraw_col = f'm{m}_paybill_total_value', f'm{m}_merchantpay_total_value', f'm{m}_withdraw_total_value'
        if paybill_col in result.columns and merchant_col in result.columns:
            result[f'm{m}_paybill_vs_merchant'] = result[paybill_col] / (result[merchant_col] + 1e-5)
            result[f'm{m}_withdraw_vs_merchant'] = result[withdraw_col] / (result[merchant_col] + 1e-5)

    # 11. Recency
    vol_cols = [f'm{m}_total_volume' for m in range(1, 7)]
    if all(c in result.columns for c in vol_cols):
        vol_matrix = result[vol_cols].values
        months_since_active = np.argmax(vol_matrix > 0, axis=1)
        months_since_active[~(vol_matrix > 0).any(axis=1)] = 6
        result['months_since_last_activity'] = months_since_active

    for c in cat_cols:
        if c in result.columns:
            result[c] = result[c].fillna("Missing").astype(str)

    return result

print("\n--- Engineering Features ---")
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# --- Peer-Relative Features ---
all_df = pd.concat([train_fe, test_fe], axis=0).reset_index(drop=True)
for col in ['segment', 'earning_pattern', 'region']:
    all_df[f'{col}_mean_arpu'] = all_df.groupby(col)['arpu'].transform('mean')
    all_df[f'arpu_vs_{col}_peers'] = all_df['arpu'] / (all_df[f'{col}_mean_arpu'] + 1e-5)
    all_df[f'{col}_mean_m1_bal'] = all_df.groupby(col)['m1_daily_avg_bal'].transform('mean')
    all_df[f'm1_bal_vs_{col}_peers'] = all_df['m1_daily_avg_bal'] / (all_df[f'{col}_mean_m1_bal'] + 1e-5)

train_fe = all_df.iloc[:len(train_fe)].copy()
test_fe = all_df.iloc[len(train_fe):].copy()

# --- K-Fold Target Encoding ---
print("--- Applying K-Fold Target Encoding ---")
train_fe[TARGET] = y
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
for c in cat_cols:
    train_fe[f'{c}_te'], test_fe[f'{c}_te'] = np.nan, np.nan

for tr_idx, val_idx in kf_te.split(train_fe):
    X_tr, X_val = train_fe.iloc[tr_idx], train_fe.iloc[val_idx]
    for c in cat_cols:
        train_fe.loc[val_idx, f'{c}_te'] = X_val[c].map(X_tr.groupby(c)[TARGET].mean())

for c in cat_cols:
    test_fe[f'{c}_te'] = test_fe[c].map(train_fe.groupby(c)[TARGET].mean())
    train_fe[f'{c}_te'].fillna(0.1500, inplace=True)
    test_fe[f'{c}_te'].fillna(0.1500, inplace=True)

# --- Feature Selection ---
INITIAL_FEATURES = [c for c in train_fe.columns if c not in [TARGET, "profile_hash"]]

print("--- Running Feature Selection ---")
temp_X = train_fe[INITIAL_FEATURES].copy()
for c in cat_cols:
    if c in temp_X.columns: temp_X[c] = temp_X[c].astype('category')

temp_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
temp_model.fit(temp_X, y)

importance_df = pd.DataFrame({'Feature': INITIAL_FEATURES, 'Importance': temp_model.feature_importances_})
best_features = importance_df.sort_values('Importance', ascending=False).head(200)['Feature'].tolist()

for c in cat_cols:
    if c not in best_features and c in INITIAL_FEATURES:
        best_features.append(c)

print(f"Reduced features from {len(INITIAL_FEATURES)} down to {len(best_features)}")
FEATURE_COLS = best_features 
groups = train_fe["profile_hash"].values
cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in cat_cols]

# --- Label Encoding ---
X = train_fe[FEATURE_COLS].values
test_X = test_fe[FEATURE_COLS].values

for i, c in enumerate(FEATURE_COLS):
    if c in cat_cols and c in train_fe.columns:
        le = LabelEncoder()
        all_vals = np.unique(np.concatenate([train_fe[c].astype(str).values, test_fe[c].astype(str).values]))
        le.fit(all_vals)
        X[:, i] = le.transform(train_fe[c].astype(str).values)
        test_X[:, i] = le.transform(test_fe[c].astype(str).values)

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups))

def train_oof(model_name, model_fn, model_params, X, y, test_X, splits, cat_features=None):
    oof_preds, test_preds = np.zeros(len(X)), np.zeros(len(test_X))
    cat_features = cat_features or []
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_fn(**model_params)

        if model_name == "lgb":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], categorical_feature=cat_features, callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
        elif model_name == "xgb":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "cb":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], cat_features=cat_features, verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(test_X)[:, 1] / len(splits)

    score = log_loss(y, oof_preds)
    print(f"Overall {model_name.upper()} LogLoss: {score:.5f}")
    return oof_preds, test_preds

# =========================================================
# THE GRANDMASTER AUTOMATED MULTI-SEED LOOP
# =========================================================
SEEDS = [42, 999]
all_seed_predictions = []

if not os.path.exists("submissions"):
    os.makedirs("submissions")

for seed in SEEDS:
    print(f"\n=========================================")
    print(f"   🚀 TRAINING ENSEMBLE WITH SEED {seed}   ")
    print(f"=========================================")
    seed_everything(seed)
    
    lgb_params = {
        "n_estimators": 10000, "learning_rate": 0.005, "max_depth": 12, "num_leaves": 34,
        "subsample": 0.7153, "colsample_bytree": 0.5692, "min_child_samples": 35,
        "reg_alpha": 0.0435, "reg_lambda": 1.5197, "random_state": seed, "verbose": -1,
        "objective": "binary", "metric": "binary_logloss"
    }
    oof_lgb, test_lgb = train_oof("lgb", lgb.LGBMClassifier, lgb_params, X, y, test_X, splits, cat_features=cat_indices)

    xgb_params = {
        "n_estimators": 10000, "learning_rate": 0.005, "max_depth": 9, "subsample": 0.8681,
        "colsample_bytree": 0.6235, "min_child_weight": 25, "reg_alpha": 0.2149, "reg_lambda": 0.2036,
        "random_state": seed, "verbosity": 0, "early_stopping_rounds": 300 
    }
    oof_xgb, test_xgb = train_oof("xgb", xgb.XGBClassifier, xgb_params, X, y, test_X, splits)

    cb_params = {
        "iterations": 10000, "learning_rate": 0.008, "depth": 8, "l2_leaf_reg": 16.3477,
        "colsample_bylevel": 0.9540, "border_count": 512, "random_seed": seed, "early_stopping_rounds": 300, "verbose": 0,
    }
    oof_cb, test_cb = train_oof("cb", CatBoostClassifier, cb_params, X, y, test_X, splits, cat_features=cat_indices)

    # Calculate Weighted OOF for this Seed
    weighted_oof = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cb
    w_ll = log_loss(y, weighted_oof)
    w_auc = roc_auc_score(y, weighted_oof)
    print(f"\n---> SEED {seed} | Weighted OOF LogLoss: {w_ll:.5f} | ROC-AUC: {w_auc:.5f}")

    # Generate Test Predictions for this Seed
    seed_test_preds = 0.4 * test_lgb + 0.3 * test_xgb + 0.3 * test_cb
    all_seed_predictions.append(seed_test_preds)
    
    # Save log
    log = pd.DataFrame({"timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "seed": [seed], "logloss": [w_ll], "roc_auc": [w_auc]})
    log.to_csv("submissions/cv_log.csv", mode="a", header=not os.path.exists("submissions/cv_log.csv"), index=False)

# =========================================================
# FINAL BLEND & SUBMISSION
# =========================================================
print("\n=========================================")
print("   🧬 BLENDING SEEDS & SAVING FILE...   ")
print("=========================================")

final_blend = np.mean(all_seed_predictions, axis=0)
final_blend = np.clip(final_blend, 1e-5, 1 - 1e-5)

sub["Target"] = final_blend
sub.to_csv("submissions/submission_ULTIMATE_blend.csv", index=False)

print("SUCCESS! File saved as 'submissions/submission_ULTIMATE_blend.csv'")
