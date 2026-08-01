import os, random, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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

# --- GLOBAL SEED CONTROL ---
# Change this to 999 after your first run!
CURRENT_SEED = 999
seed_everything(CURRENT_SEED)

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

y = train[TARGET].values
cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]

def engineer_features(df, is_train=True):
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if is_train and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    if is_train:
        df["profile_hash"] = (
                df["age"].astype(str) + "_" +
                df["gender"].astype(str) + "_" +
                df["region"].astype(str)
        )

    temporal_metrics = {}
    balance_cols = []
    for col in df.columns:
        if re.match(r"^m[1-6]_daily_avg_bal$", col):
            balance_cols.append(col)
        m = re.match(r"^m([1-6])_([a-z_]+)_(volume|total_value|highest_amount)$", col)
        if m:
            month, txn_type, metric = m.groups()
            key = (txn_type, metric)
            temporal_metrics.setdefault(key, {})[int(month)] = col
        m2 = re.match(r"^m([1-6])_([a-z_]+)_([a-z_]+)$", col)
        if m2:
            month, txn_type, entity = m2.groups()
            if entity in ("companies", "merchants", "banks", "recipients", "senders", "agents") and month in "123456" and txn_type in ("paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"):
                key = (txn_type, entity)
                temporal_metrics.setdefault(key, {})[int(month)] = col

    new_features = {}
    for key, month_map in temporal_metrics.items():
        vals = []
        for m in range(1, 7):
            c = month_map.get(m)
            vals.append(df[c].values if c else np.zeros(len(df)))
        vals = np.column_stack(vals)
        months = np.arange(1, 7).astype(float)
        
        slopes = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in vals])
        means = vals.mean(axis=1)
        stds = vals.std(axis=1)
        
        txn_type, metric = key
        prefix = f"{txn_type}_{metric}"
        
        weights = np.array([6, 5, 4, 3, 2, 1])
        ema = np.average(vals, axis=1, weights=weights)
        new_features[f"{prefix}_ema"] = ema
        
        recent = vals[:, 0]
        oldest = vals[:, 5]
        ratio = np.divide(recent, oldest, out=np.zeros_like(recent, dtype=float), where=oldest != 0)
        zeros = (vals == 0).sum(axis=1)
        new_features[f"{prefix}_slope"] = slopes
        new_features[f"{prefix}_mean"] = means
        new_features[f"{prefix}_std"] = stds 
        new_features[f"{prefix}_recent_old_ratio"] = ratio
        new_features[f"{prefix}_dormant_months"] = zeros

        for m in range(5):
            delta = vals[:, m] - vals[:, m + 1]
            new_features[f"{prefix}_delta_m{m+1}"] = delta

    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        months = np.arange(1, 7).astype(float)
        bal_slopes = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in bal_vals])
        new_features["bal_slope"] = bal_slopes
        new_features["bal_mean"] = bal_vals.mean(axis=1)
        new_features["bal_volatility"] = bal_vals.std(axis=1)
        new_features["bal_min"] = bal_vals.min(axis=1)
        new_features["bal_decline"] = (bal_vals[:, 0] < bal_vals[:, 5]).astype(float)

    txn_types = ["paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw"]
    for m in range(1, 7):
        vol_list = []
        for t in txn_types:
            c = f"m{m}_{t}_volume"
            if c in df.columns:
                vol_list.append(df[c].values)
        if vol_list:
            total_vol = np.column_stack(vol_list).sum(axis=1)
            new_features[f"m{m}_total_volume"] = total_vol

        val_list = []
        for t in txn_types:
            c = f"m{m}_{t}_total_value"
            if c in df.columns:
                val_list.append(df[c].values)
        if val_list:
            total_val = np.column_stack(val_list).sum(axis=1)
            new_features[f"m{m}_total_value"] = total_val

    for m in range(1, 7):
        w_col = f"m{m}_withdraw_total_value"
        d_col = f"m{m}_deposit_total_value"
        if w_col in df.columns and d_col in df.columns:
            ratio_wd = np.divide(df[w_col].values, df[d_col].values, out=np.zeros(len(df)), where=df[d_col].values != 0)
            new_features[f"m{m}_wd_ratio"] = ratio_wd

    new_features['arpu_to_m1_bal'] = df['arpu'] / (df['m1_daily_avg_bal'] + 1e-5)
    new_features['arpu_to_m1_withdraw'] = df['arpu'] / (df['m1_withdraw_total_value'] + 1e-5)
    new_features['x_90_d_activity_to_arpu'] = df['x_90_d_activity_rate'] / (df['arpu'] + 1e-5)

    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        short_term_bal = bal_vals[:, 0:2].mean(axis=1)
        long_term_bal = bal_vals[:, 3:6].mean(axis=1)
        new_features['bal_macd_ratio'] = short_term_bal / (long_term_bal + 1e-5)
        
    for prefix in ['deposit_total_value', 'withdraw_total_value', 'received_total_value']:
        m1_col, m2_col, m4_col, m5_col, m6_col = f'm1_{prefix}', f'm2_{prefix}', f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}'
        if all(c in df.columns for c in [m1_col, m2_col, m4_col, m5_col, m6_col]):
            short_term = df[[m1_col, m2_col]].mean(axis=1)
            long_term = df[[m4_col, m5_col, m6_col]].mean(axis=1)
            new_features[f'{prefix}_macd_ratio'] = short_term / (long_term + 1e-5)

    inflow_types = ['deposit', 'received', 'transfer_from_bank']
    outflow_types = ['withdraw', 'merchantpay', 'paybill', 'mm_send']

    for m in range(1, 7):
        inflow = np.zeros(len(df))
        outflow = np.zeros(len(df))
        for t in inflow_types:
            col = f'm{m}_{t}_total_value'
            if col in df.columns: inflow += df[col].values
        for t in outflow_types:
            col = f'm{m}_{t}_total_value'
            if col in df.columns: outflow += df[col].values

        new_features[f'm{m}_net_cashflow'] = inflow - outflow
        new_features[f'm{m}_cashflow_margin'] = (inflow - outflow) / (inflow + 1e-5)
        new_features[f'm{m}_cashflow_to_arpu'] = new_features[f'm{m}_net_cashflow'] / (df['arpu'] + 1e-5)

    if 'm1_daily_avg_bal' in df.columns:
        new_features['is_m1_bankrupt'] = (df['m1_daily_avg_bal'] == 0).astype(float)
        new_features['is_m2_bankrupt'] = (df['m2_daily_avg_bal'] == 0).astype(float)

    cashflow_matrix = np.column_stack([new_features[f'm{m}_net_cashflow'] for m in range(1, 7)])
    months = np.arange(1, 7).astype(float)
    new_features['net_cashflow_slope'] = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in cashflow_matrix])

    df_new = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, df_new], axis=1)

    # --- NEW: 90-Day Alignment Features ---
    for prefix in ['daily_avg_bal', 'net_cashflow', 'deposit_total_value', 'withdraw_total_value']:
        if f'm1_{prefix}' in result.columns:
            recent_90d = result[[f'm1_{prefix}', f'm2_{prefix}', f'm3_{prefix}']].mean(axis=1)
            prior_90d = result[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1)
            result[f'{prefix}_90d_trend'] = recent_90d / (prior_90d + 1e-5)
            result[f'{prefix}_90d_delta'] = recent_90d - prior_90d
            
    # --- NEW: Necessity vs Discretionary Spending (Desperation Signal) ---
    for m in range(1, 4):
        paybill_col = f'm{m}_paybill_total_value'
        merchant_col = f'm{m}_merchantpay_total_value'
        withdraw_col = f'm{m}_withdraw_total_value'
        if paybill_col in result.columns and merchant_col in result.columns:
            result[f'm{m}_paybill_vs_merchant'] = result[paybill_col] / (result[merchant_col] + 1e-5)
            result[f'm{m}_withdraw_vs_merchant'] = result[withdraw_col] / (result[merchant_col] + 1e-5)

    for c in cat_cols:
        if c in result.columns:
            result[c] = result[c].fillna("Missing").astype(str)

    return result

train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# Calculate Peer-Relative Features
all_df = pd.concat([train_fe, test_fe], axis=0).reset_index(drop=True)
group_cols = ['segment', 'earning_pattern', 'region']
for col in group_cols:
    all_df[f'{col}_mean_arpu'] = all_df.groupby(col)['arpu'].transform('mean')
    all_df[f'arpu_vs_{col}_peers'] = all_df['arpu'] / (all_df[f'{col}_mean_arpu'] + 1e-5)
    all_df[f'{col}_mean_m1_bal'] = all_df.groupby(col)['m1_daily_avg_bal'].transform('mean')
    all_df[f'm1_bal_vs_{col}_peers'] = all_df['m1_daily_avg_bal'] / (all_df[f'{col}_mean_m1_bal'] + 1e-5)

train_fe = all_df.iloc[:len(train_fe)].copy()
test_fe = all_df.iloc[len(train_fe):].copy()

# --- NEW: K-Fold Target Encoding ---
print("\n--- Applying K-Fold Target Encoding ---")
train_fe[TARGET] = y
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
for c in cat_cols:
    train_fe[f'{c}_te'] = np.nan
    test_fe[f'{c}_te'] = np.nan

for tr_idx, val_idx in kf_te.split(train_fe):
    X_tr, X_val = train_fe.iloc[tr_idx], train_fe.iloc[val_idx]
    for c in cat_cols:
        target_means = X_tr.groupby(c)[TARGET].mean()
        train_fe.loc[val_idx, f'{c}_te'] = X_val[c].map(target_means)

for c in cat_cols:
    global_means = train_fe.groupby(c)[TARGET].mean()
    test_fe[f'{c}_te'] = test_fe[c].map(global_means)
    train_fe[f'{c}_te'] = train_fe[f'{c}_te'].fillna(0.1500)
    test_fe[f'{c}_te'] = test_fe[f'{c}_te'].fillna(0.1500)


INITIAL_FEATURES = [c for c in train_fe.columns if c not in [TARGET, "profile_hash"]]

print("\n--- Running Feature Selection ---")
temp_X = train_fe[INITIAL_FEATURES].copy()
for c in cat_cols:
    if c in temp_X.columns:
        temp_X[c] = temp_X[c].astype('category')

temp_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
temp_model.fit(temp_X, y)

importance_df = pd.DataFrame({'Feature': INITIAL_FEATURES, 'Importance': temp_model.feature_importances_})
importance_df = importance_df.sort_values('Importance', ascending=False)
best_features = importance_df.head(200)['Feature'].tolist() # BUMPED TO 200

for c in cat_cols:
    if c not in best_features and c in INITIAL_FEATURES:
        best_features.append(c)

print(f"Reduced features from {len(INITIAL_FEATURES)} down to {len(best_features)}")

FEATURE_COLS = best_features 
X = train_fe[FEATURE_COLS].values
test_X = test_fe[FEATURE_COLS].values
groups = train_fe["profile_hash"].values

cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in cat_cols]

le_dict = {}
for i, c in enumerate(FEATURE_COLS):
    if c in cat_cols and c in train_fe.columns:
        le = LabelEncoder()
        train_vals = train_fe[c].astype(str).values
        test_vals = test_fe[c].astype(str).values
        all_vals = np.unique(np.concatenate([train_vals, test_vals]))
        le.fit(all_vals)
        X[:, i] = le.transform(train_vals)
        test_X[:, i] = le.transform(test_vals)
        le_dict[c] = le

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups))

def train_oof(model_name, model_fn, model_params, X, y, test_X, splits, cat_features=None):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_X))
    models = []

    if cat_features is None:
        cat_features = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = model_fn(**model_params)

        if model_name == "lgb":
            model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    categorical_feature=cat_features,
                    callbacks=[lgb.early_stopping(100, verbose=False),lgb.log_evaluation(0)]
                    )
        elif model_name == "xgb":
            model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                    )
        elif model_name == "cb":
            model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    cat_features=cat_features,
                    verbose=False
                    )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs

        test_preds += model.predict_proba(test_X)[:, 1] / len(splits)
        models.append(model)

        score = log_loss(y_val, val_probs)
        auc = roc_auc_score(y_val, val_probs)
        print(f" Fold {fold+1} | Logloss: {score:.5f} | ROC-AUC: {auc:.5f}")

    return oof_preds, test_preds, models

print("\n--- LightGBM ---")
lgb_params = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": CURRENT_SEED,
    "verbose": -1,
    "objective": "binary",
    "metric": "binary_logloss"
}
oof_lgb, test_lgb, models_lgb = train_oof(
    "lgb", lgb.LGBMClassifier, lgb_params, X, y, test_X, splits
)

print("\n--- XGBoost ---")
xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": CURRENT_SEED,
    "verbosity": 0,
    "early_stopping_rounds": 100 
}
oof_xgb, test_xgb, models_xgb = train_oof(
    "xgb", xgb.XGBClassifier, xgb_params, X, y, test_X, splits
)

print("\n--- CatBoost ---")
cb_params = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 10,
    "random_seed": CURRENT_SEED,
    "early_stopping_rounds": 100,
    "verbose": 0,
}
oof_cb, test_cb, models_cb = train_oof(
    "cb", CatBoostClassifier, cb_params, X, y, test_X, splits, cat_features=cat_indices
)

print("\n--- Stacking with LogisticRegression ---")
stack_features = np.column_stack([oof_lgb, oof_xgb, oof_cb])
stack_model = LogisticRegression(C=0.01, solver="lbfgs", random_state=CURRENT_SEED, max_iter=1000)
stack_model.fit(stack_features, y)

stack_oof = stack_model.predict_proba(stack_features)[:, 1]
stack_ll = log_loss(y, stack_oof)
stack_auc = roc_auc_score(y, stack_oof)
print(f"Stack OOF | LogLoss: {stack_ll:.5f} | ROC-AUC: {stack_auc:.5f}")

weighted_oof = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cb
w_ll = log_loss(y, weighted_oof)
w_auc = roc_auc_score(y, weighted_oof)
print(f"Weighted OOF | LogLoss: {w_ll:.5f} | ROC-AUC: {w_auc:.5f}")

log = pd.DataFrame({
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 2,
    "model": ["stack", "weighted"],
    "logloss": [stack_ll, w_ll],
    "roc_auc": [stack_auc, w_auc],
    "features": [len(FEATURE_COLS), len(FEATURE_COLS)]})
if not os.path.exists("submissions"):
    os.makedirs("submissions")
log.to_csv("submissions/cv_log.csv", mode="a", header=not os.path.exists("submissions/cv_log.csv"), index=False)

print("\n--- Generating test predictions ---")
test_stack = np.column_stack([test_lgb, test_xgb, test_cb])
final_stack_preds = stack_model.predict_proba(test_stack)[:, 1]
final_weight_preds = 0.4 * test_lgb + 0.3 * test_xgb + 0.3 * test_cb

final_weight_preds = np.clip(final_weight_preds, 1e-5, 1 - 1e-5)

sub["Target"] = final_stack_preds
sub.to_csv(f"submissions/submission_stacked_seed_{CURRENT_SEED}.csv", index=False)

sub["Target"] = final_weight_preds
sub.to_csv(f"submissions/submission_seed_{CURRENT_SEED}.csv", index=False)

print(f"Saved predictions for SEED {CURRENT_SEED}!")

print("\n--- Top 20 Most Important Features (CatBoost) ---")
importance = models_cb[0].get_feature_importance()
feat_imp = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': importance
}).sort_values('Importance', ascending=False)
print(feat_imp.head(20))
