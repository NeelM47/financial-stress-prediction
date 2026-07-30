import os, random, warnings, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
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

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"

y = train[TARGET].values
print("Target mean: {:.4f}".format(y.mean()))
print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))

cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]
#print("Missing in train:\n", train.isnull().sum().to_string())

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
        recent = vals[:, 0]
        oldest = vals[:, 5]
        ratio = np.divide(recent, oldest, out=np.zeros_like(recent, dtype=float), where=oldest != 0)
        zeros = (vals == 0).sum(axis=1)
        txn_type, metric = key
        prefix = f"{txn_type}_{metric}"
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

# 1. ARPU Interactions (Add 1e-5 to avoid division by zero)
    new_features['arpu_to_m1_bal'] = df['arpu'] / (df['m1_daily_avg_bal'] + 1e-5)
    new_features['arpu_to_m1_withdraw'] = df['arpu'] / (df['m1_withdraw_total_value'] + 1e-5)
    new_features['x_90_d_activity_to_arpu'] = df['x_90_d_activity_rate'] / (df['arpu'] + 1e-5)

    # 2. Short-Term (Last 2 months) vs Long-Term (Oldest 3 months) Velocity
    if balance_cols:
        bal_vals = df[balance_cols].values.astype(float)
        short_term_bal = bal_vals[:, 0:2].mean(axis=1) # M1 (most recent) and M2
        long_term_bal = bal_vals[:, 3:6].mean(axis=1)  # M4, M5, M6 (oldest)
        new_features['bal_macd_ratio'] = short_term_bal / (long_term_bal + 1e-5)
    for prefix in ['deposit_total_value', 'withdraw_total_value', 'received_total_value']:
        m1_col, m2_col = f'm1_{prefix}', f'm2_{prefix}'
        m4_col, m5_col, m6_col = f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}'
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
            if col in df.columns:
                inflow += df[col].values

        for t in outflow_types:
            col = f'm{m}_{t}_total_value'
            if col in df.columns:
                outflow += df[col].values

        new_features[f'm{m}_net_cashflow'] = inflow - outflow

        new_features[f'm{m}_cashflow_margin'] = (inflow - outflow) / (inflow + 1e-5)
        # 3. Cashflow relative to historical revenue
        new_features[f'm{m}_cashflow_to_arpu'] = new_features[f'm{m}_net_cashflow'] / (df['arpu'] + 1e-5)

    if 'm1_daily_avg_bal' in df.columns:
        new_features['is_m1_bankrupt'] = (df['m1_daily_avg_bal'] == 0).astype(float)
        new_features['is_m2_bankrupt'] = (df['m2_daily_avg_bal'] == 0).astype(float)

    cashflow_matrix = np.column_stack([new_features[f'm{m}_net_cashflow'] for m in range(1, 7)])
    months = np.arange(1, 7).astype(float)
    new_features['net_cashflow_slope'] = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in cashflow_matrix])

    df_new = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, df_new], axis=1)

    # FIX 1: Do NOT drop the raw M1-M6 temporal features! 
    # Tree models thrive on raw thresholds.
    # drop_pattern = re.compile(r"^m[1-6]_")
    # drop_cols = [c for c in result.columns if drop_pattern.match(c)]
    # result = result.drop(columns=drop_cols)

    for c in cat_cols:
        if c in result.columns:
            result[c] = result[c].fillna("Missing").astype(str)

    return result

train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# Combine temporarily to calculate highly accurate group statistics
all_df = pd.concat([train_fe, test_fe], axis=0).reset_index(drop=True)

# Calculate Peer-Relative Features
group_cols = ['segment', 'earning_pattern', 'region']
for col in group_cols:
    # Mean ARPU per demographic group
    all_df[f'{col}_mean_arpu'] = all_df.groupby(col)['arpu'].transform('mean')
    all_df[f'arpu_vs_{col}_peers'] = all_df['arpu'] / (all_df[f'{col}_mean_arpu'] + 1e-5)
    
    # Mean Month 1 Balance per demographic group
    all_df[f'{col}_mean_m1_bal'] = all_df.groupby(col)['m1_daily_avg_bal'].transform('mean')
    all_df[f'm1_bal_vs_{col}_peers'] = all_df['m1_daily_avg_bal'] / (all_df[f'{col}_mean_m1_bal'] + 1e-5)

# Split back into train and test flawlessly
train_fe = all_df.iloc[:len(train_fe)].copy()
test_fe = all_df.iloc[len(train_fe):].copy()

FEATURE_COLS = [c for c in train_fe.columns if c not in [TARGET, "profile_hash"]]
X = train_fe[FEATURE_COLS].values
test_X = test_fe[FEATURE_COLS].values
groups = train_fe["profile_hash"].values

# FIX 4: Find categorical indices so CatBoost can handle them natively
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

# FIX 2 & 3: Rewrite CV loop to include Early Stopping and accumulate test predictions per fold
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
    "random_state": 42,
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
    "random_state": 42,
    "verbosity": 0,
    "early_stopping_rounds": 100 # Added to params for XGB
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
    "random_seed": 42,
    "early_stopping_rounds": 100, # Added for Catboost
    "verbose": 0,
}
oof_cb, test_cb, models_cb = train_oof(
    "cb", CatBoostClassifier, cb_params, X, y, test_X, splits, cat_features=cat_indices
)

print("\n--- Stacking with LogisticRegression ---")
stack_features = np.column_stack([oof_lgb, oof_xgb, oof_cb])
stack_model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42, max_iter=1000)
stack_model.fit(stack_features, y)

stack_oof = stack_model.predict_proba(stack_features)[:, 1]
stack_ll = log_loss(y, stack_oof)
stack_auc = roc_auc_score(y, stack_oof)
print(f"Stack OOF | LogLoss: {stack_ll:.5f} | ROC-AUC: {stack_auc:.5f}")

weighted_oof = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cb
w_ll = log_loss(y, weighted_oof)
w_auc = roc_auc_score(y, weighted_oof)
print(f"Weighted OOF | LogLoss: {w_ll:.5f} | ROC-AUC: {w_auc:.5f}")

print("\n--- Generating test predictions ---")

# FIX 3: We no longer do a full retrain! 

# We use the accumulated test predictions directly fed into the stacker.
test_stack = np.column_stack([test_lgb, test_xgb, test_cb])
final_stack_preds = stack_model.predict_proba(test_stack)[:, 1]

# Optional Weighted Predictions
final_weight_preds = 0.4 * test_lgb + 0.3 * test_xgb + 0.3 * test_cb

final_weight_preds = np.clip(final_weight_preds, 1e-5, 1 - 1e-5)

# We will use the Stacker's predictions for submission as it mathematically optimizes LogLoss
sub["Target"] = final_stack_preds
sub.to_csv("submission_stacked.csv", index=False)

# Just in case, let's also save the weighted predictions
sub["Target"] = final_weight_preds
sub.to_csv("submission_weighted_safe.csv", index=False)

print("Saved submission_stacked.csv and submission_weighted_safe.csv - shape:", sub.shape)
#print("Min Pred:", final_weight_preds.min(), "| Max Pred:", final_weight_preds.max())
#print(sub.head())

print("\n--- Top 20 Most Important Features (CatBoost) ---")
# models_cb[0] is the CatBoost model trained on Fold 1
importance = models_cb[0].get_feature_importance()
feat_imp = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(feat_imp.head(20))
