import os, random, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb  # Used ONLY for fast feature pruning
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# ==========================================
# 1. LOAD DATA 
# ==========================================
print("Loading data...")
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
y = train[TARGET].values

cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern", "demo_fingerprint"]

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df, is_train=True):
    df = df.copy()
    df['demo_fingerprint'] = df['region'].astype(str) + "_" + df['segment'].astype(str) + "_" + df['earning_pattern'].astype(str)
    
    if ID_COL in df.columns:
        df['id_length'] = df[ID_COL].apply(len)
        df['id_numeric'] = df[ID_COL].apply(lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else 0)
        df = df.drop(columns=[ID_COL])
    if is_train and TARGET in df.columns: df = df.drop(columns=[TARGET])
    if is_train: df["profile_hash"] = df["age"].astype(str) + "_" + df["gender"].astype(str) + "_" + df["region"].astype(str)

    temporal_metrics, balance_cols = {}, []
    for col in df.columns:
        if re.match(r"^m[1-6]_daily_avg_bal$", col): balance_cols.append(col)
        m = re.match(r"^m([1-6])_([a-z_]+)_(volume|total_value|highest_amount)$", col)
        if m: temporal_metrics.setdefault((m.group(2), m.group(3)), {})[int(m.group(1))] = col
        m2 = re.match(r"^m([1-6])_([a-z_]+)_([a-z_]+)$", col)
        if m2 and m2.group(3) in ("companies", "merchants", "banks", "recipients", "senders", "agents"):
            temporal_metrics.setdefault((m2.group(2), m2.group(3)), {})[int(m2.group(1))] = col

    new_features = {}
    for key, month_map in temporal_metrics.items():
        vals = np.column_stack([df[month_map.get(m)].values if month_map.get(m) else np.zeros(len(df)) for m in range(1, 7)])
        months = np.arange(1, 7).astype(float)
        txn_type, metric = key
        prefix = f"{txn_type}_{metric}"
        new_features[f"{prefix}_ema"] = np.average(vals, axis=1, weights=np.array([6, 5, 4, 3, 2, 1]))
        new_features[f"{prefix}_slope"] = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in vals])
        new_features[f"{prefix}_mean"] = vals.mean(axis=1)
        new_features[f"{prefix}_std"] = vals.std(axis=1)
        new_features[f"{prefix}_recent_old_ratio"] = np.divide(vals[:, 0], vals[:, 5], out=np.zeros_like(vals[:, 0], dtype=float), where=vals[:, 5] != 0)
        new_features[f"{prefix}_dormant_months"] = (vals == 0).sum(axis=1)
        for m in range(5): new_features[f"{prefix}_delta_m{m+1}"] = vals[:, m] - vals[:, m + 1]

    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        new_features["bal_slope"] = np.array([np.polyfit(np.arange(1, 7).astype(float), v, 1)[0] if np.any(v != 0) else 0.0 for v in bal_vals])
        new_features["bal_mean"] = bal_vals.mean(axis=1)
        new_features["bal_volatility"] = bal_vals.std(axis=1)
        new_features["bal_min"] = bal_vals.min(axis=1)
        new_features["bal_decline"] = (bal_vals[:, 0] < bal_vals[:, 5]).astype(float)

    txn_types = ["paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"]
    for m in range(1, 7):
        vol_list = [df[f"m{m}_{t}_volume"].values for t in txn_types if f"m{m}_{t}_volume" in df.columns]
        if vol_list: new_features[f"m{m}_total_volume"] = np.column_stack(vol_list).sum(axis=1)
        val_list = [df[f"m{m}_{t}_total_value"].values for t in txn_types if f"m{m}_{t}_total_value" in df.columns]
        if val_list: new_features[f"m{m}_total_value"] = np.column_stack(val_list).sum(axis=1)
        if f"m{m}_withdraw_total_value" in df.columns and f"m{m}_deposit_total_value" in df.columns:
            new_features[f"m{m}_wd_ratio"] = np.divide(df[f"m{m}_withdraw_total_value"].values, df[f"m{m}_deposit_total_value"].values, out=np.zeros(len(df)), where=df[f"m{m}_deposit_total_value"].values != 0)

    for m in range(1, 7):
        for t in txn_types:
            vol_col = f"m{m}_{t}_volume"
            val_col = f"m{m}_{t}_total_value" 
            if vol_col in df.columns and val_col in df.columns:
                new_features[f"m{m}_{t}_avg_size"] = np.divide(df[val_col].values, df[vol_col].values, out=np.zeros(len(df)), where=df[vol_col].values != 0 )

    if all(f'm{m}_withdraw_avg_size' in new_features for m in range(1, 7)):
        recent_size = np.mean([new_features[f'm{m}_withdraw_avg_size'] for m in range(1, 4)], axis=0)
        old_size = np.mean([new_features[f'm{m}_withdraw_avg_size'] for m in range(4, 7)], axis=0)
        new_features['withdraw_avg_size_90d_trend'] = recent_size / (old_size + 1e-5)

    new_features['arpu_to_m1_bal'] = df['arpu'] / (df['m1_daily_avg_bal'] + 1e-5)
    new_features['x_90_d_activity_to_arpu'] = df['x_90_d_activity_rate'] / (df['arpu'] + 1e-5)

    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        new_features['bal_macd_ratio'] = bal_vals[:, 0:2].mean(axis=1) / (bal_vals[:, 3:6].mean(axis=1) + 1e-5)
        
    for prefix in ['deposit_total_value', 'withdraw_total_value', 'received_total_value']:
        if all(f'm{i}_{prefix}' in df.columns for i in [1, 2, 4, 5, 6]):
            new_features[f'{prefix}_macd_ratio'] = df[[f'm1_{prefix}', f'm2_{prefix}']].mean(axis=1) / (df[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1) + 1e-5)

    inflow_types = ['deposit', 'received', 'transfer_from_bank']
    outflow_types = ['withdraw', 'merchantpay', 'paybill', 'mm_send']

    for m in range(1, 7):
        inflow = sum([df[f'm{m}_{t}_total_value'].values for t in inflow_types if f'm{m}_{t}_total_value' in df.columns])
        outflow = sum([df[f'm{m}_{t}_total_value'].values for t in outflow_types if f'm{m}_{t}_total_value' in df.columns])
        if type(inflow) != int:
            new_features[f'm{m}_net_cashflow'] = inflow - outflow
            new_features[f'm{m}_cashflow_margin'] = (inflow - outflow) / (inflow + 1e-5)

    cashflow_matrix = np.column_stack([new_features[f'm{m}_net_cashflow'] for m in range(1, 7)])
    new_features['net_cashflow_slope'] = np.array([np.polyfit(np.arange(1, 7).astype(float), v, 1)[0] if np.any(v != 0) else 0.0 for v in cashflow_matrix])

    df_new = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, df_new], axis=1)

    for prefix in ['daily_avg_bal', 'net_cashflow', 'deposit_total_value', 'withdraw_total_value']:
        if f'm1_{prefix}' in result.columns:
            recent_90d = result[[f'm1_{prefix}', f'm2_{prefix}', f'm3_{prefix}']].mean(axis=1)
            prior_90d = result[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1)
            result[f'{prefix}_90d_trend'] = recent_90d / (prior_90d + 1e-5)
            result[f'{prefix}_90d_delta'] = recent_90d - prior_90d

    vol_cols = [f'm{m}_total_volume' for m in range(1, 7)]
    if all(c in result.columns for c in vol_cols):
        vol_matrix = result[vol_cols].values
        months_since_active = np.argmax(vol_matrix > 0, axis=1)
        months_since_active[~(vol_matrix > 0).any(axis=1)] = 6
        result['months_since_last_activity'] = months_since_active

    for c in cat_cols:
        if c in result.columns: result[c] = result[c].fillna("Missing").astype(str)
    return result

print("\n--- Engineering Features ---")
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# Peer-Relative Baseline
all_df = pd.concat([train_fe, test_fe], axis=0).reset_index(drop=True)
for col in ['segment', 'earning_pattern', 'region']:
    all_df[f'{col}_mean_arpu'] = all_df.groupby(col)['arpu'].transform('mean')
    all_df[f'arpu_vs_{col}_peers'] = all_df['arpu'] / (all_df[f'{col}_mean_arpu'] + 1e-5)
train_fe = all_df.iloc[:len(train_fe)].copy()
test_fe = all_df.iloc[len(train_fe):].copy()

# K-Fold Target Encoding
print("--- Applying K-Fold Target Encoding ---")
train_fe[TARGET] = y
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
for c in cat_cols: train_fe[f'{c}_te'], test_fe[f'{c}_te'] = np.nan, np.nan
for tr_idx, val_idx in kf_te.split(train_fe):
    X_tr, X_val = train_fe.iloc[tr_idx], train_fe.iloc[val_idx]
    for c in cat_cols: train_fe.loc[val_idx, f'{c}_te'] = X_val[c].map(X_tr.groupby(c)[TARGET].mean())
for c in cat_cols:
    test_fe[f'{c}_te'] = test_fe[c].map(train_fe.groupby(c)[TARGET].mean())
    train_fe[f'{c}_te'] = train_fe[f'{c}_te'].fillna(0.1500)
    test_fe[f'{c}_te'] = test_fe[f'{c}_te'].fillna(0.1500)

# Feature Pruning
INITIAL_FEATURES = [c for c in train_fe.columns if c not in [TARGET, "profile_hash"]]
print("--- Running Feature Selection ---")
temp_X = train_fe[INITIAL_FEATURES].copy()
for c in cat_cols: temp_X[c] = temp_X[c].astype('category')
temp_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
temp_model.fit(temp_X, y)
best_features = pd.DataFrame({'Feature': INITIAL_FEATURES, 'Importance': temp_model.feature_importances_}).sort_values('Importance', ascending=False).head(200)['Feature'].tolist()
for c in cat_cols:
    if c not in best_features and c in INITIAL_FEATURES: best_features.append(c)

FEATURE_COLS = best_features 
X = train_fe[FEATURE_COLS].values
test_X = test_fe[FEATURE_COLS].values
groups = train_fe["profile_hash"].values

# Label Encoding for XGBoost
for i, c in enumerate(FEATURE_COLS):
    if c in cat_cols and c in train_fe.columns:
        le = LabelEncoder()
        all_vals = np.unique(np.concatenate([train_fe[c].astype(str).values, test_fe[c].astype(str).values]))
        le.fit(all_vals)
        X[:, i] = le.transform(train_fe[c].astype(str).values)
        test_X[:, i] = le.transform(test_fe[c].astype(str).values)

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups))

# ==========================================
# 3. XGBOOST MULTI-SEED TRAINING
# ==========================================
SEEDS = [42, 999]
all_oof_preds = []
all_test_preds = []

print("\n=========================================")
print("🚀 TRAINING XGBOOST CHAMPION MODEL 🚀")
print("=========================================")

for seed in SEEDS:
    seed_everything(seed)
    print(f"\n--- Training Seed {seed} ---")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_X))
    
    xgb_params = {
        "n_estimators": 2000, 
        "learning_rate": 0.011291, 
        "max_depth": 9, 
        "subsample": 0.8681,
        "colsample_bytree": 0.6235, 
        "min_child_weight": 25, 
        "reg_alpha": 0.2149, 
        "reg_lambda": 0.2036,
        "random_state": seed, 
        "verbosity": 0, 
        "early_stopping_rounds": 100 
    }
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(test_X)[:, 1] / len(splits)

    ll_score = log_loss(y, oof_preds)
    print(f">> Seed {seed} OOF LogLoss: {ll_score:.5f}")
    
    all_oof_preds.append(oof_preds)
    all_test_preds.append(test_preds)

# ==========================================
# 4. BLENDING & SAVING
# ==========================================
final_oof = np.mean(all_oof_preds, axis=0)
final_test = np.mean(all_test_preds, axis=0)

final_ll = log_loss(y, final_oof)
final_auc = roc_auc_score(y, final_oof)

print("\n=========================================")
print("🏆 FINAL CHAMPION PERFORMANCE 🏆")
print("=========================================")
print(f"Blended XGB OOF LOGLOSS: {final_ll:.5f}")
print(f"Blended XGB ROC-AUC:     {final_auc:.5f}")

if not os.path.exists("submissions"):
    os.makedirs("submissions")

# Safe clipping at 1e-4 prevents extreme log(0) penalties while keeping edge predictions realistic
sub["Target"] = np.clip(final_test, 1e-4, 1 - 1e-4)
sub.to_csv("submissions/xgb_champion_submission.csv", index=False)
print("\nFile saved: submissions/xgb_champion_submission.csv")
