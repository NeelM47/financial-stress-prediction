import os, random, warnings, re
import numpy as np
import pandas as pd
import optuna
import json 
from datetime import datetime
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(42)

print("Loading data...")
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
y = train[TARGET].values

cat_cols = ["gender", "region", "smartphone", "segment", "earning_pattern", "demo_fingerprint"]

def engineer_features(df, is_train=True):
    df = df.copy()
    df['demo_fingerprint'] = df['region'].astype(str) + "_" + df['segment'].astype(str) + "_" + df['earning_pattern'].astype(str)

    if ID_COL in df.columns:
        df['id_length'] = df[ID_COL].apply(len)
        df['id_numeric'] = df[ID_COL].apply(lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else 0)
        df = df.drop(columns=[ID_COL])

    if is_train and TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    if is_train:
        df["profile_hash"] = df["age"].astype(str) + "_" + df["gender"].astype(str) + "_" + df["region"].astype(str)

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
        months = np.arange(1, 7).astype(float)
        new_features["bal_slope"] = np.array([np.polyfit(months, v, 1)[0] if np.any(v != 0) else 0.0 for v in bal_vals])
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

    new_features['arpu_to_m1_bal'] = df['arpu'] / (df['m1_daily_avg_bal'] + 1e-5)
    new_features['arpu_to_m1_withdraw'] = df['arpu'] / (df['m1_withdraw_total_value'] + 1e-5)
    new_features['x_90_d_activity_to_arpu'] = df['x_90_d_activity_rate'] / (df['arpu'] + 1e-5)

    if balance_cols:
        new_features['bal_macd_ratio'] = bal_vals[:, 0:2].mean(axis=1) / (bal_vals[:, 3:6].mean(axis=1) + 1e-5)

    for prefix in ['deposit_total_value', 'withdraw_total_value', 'received_total_value']:
        if all(f'm{i}_{prefix}' in df.columns for i in [1, 2, 3, 4, 5, 6]):
            new_features[f'{prefix}_macd_ratio'] = df[[f'm1_{prefix}', f'm2_{prefix}']].mean(axis=1) / (df[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1) + 1e-5)

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

    df_new = pd.DataFrame(new_features, index=df.index)
    result = pd.concat([df, df_new], axis=1)

    for prefix in ['daily_avg_bal', 'net_cashflow', 'deposit_total_value', 'withdraw_total_value']:
        if f'm1_{prefix}' in result.columns:
            recent_90d = result[[f'm1_{prefix}', f'm2_{prefix}', f'm3_{prefix}']].mean(axis=1)
            prior_90d = result[[f'm4_{prefix}', f'm5_{prefix}', f'm6_{prefix}']].mean(axis=1)
            result[f'{prefix}_90d_trend'] = recent_90d / (prior_90d + 1e-5)
            result[f'{prefix}_90d_delta'] = recent_90d - prior_90d

    for m in range(1, 4):
        p_col, m_col, w_col = f'm{m}_paybill_total_value', f'm{m}_merchantpay_total_value', f'm{m}_withdraw_total_value'
        if p_col in result.columns and m_col in result.columns:
            result[f'm{m}_paybill_vs_merchant'] = result[p_col] / (result[m_col] + 1e-5)
            result[f'm{m}_withdraw_vs_merchant'] = result[w_col] / (result[m_col] + 1e-5)

    vol_cols = [f'm{m}_total_volume' for m in range(1, 7)]
    if all(c in result.columns for c in vol_cols):
        vol_matrix = result[vol_cols].values
        months_since_active = np.argmax(vol_matrix > 0, axis=1)
        months_since_active[~(vol_matrix > 0).any(axis=1)] = 6
        result['months_since_last_activity'] = months_since_active

    for c in cat_cols:  # demo_fingerprint etc.
        if c in result.columns: result[c] = result[c].fillna("Missing").astype(str)
    return result

print("\n--- Engineering Features ---")
train_fe = engineer_features(train, is_train=True)

all_df = train_fe.copy()
for col in ['segment', 'earning_pattern', 'region']:
    all_df[f'{col}_mean_arpu'] = all_df.groupby(col)['arpu'].transform('mean')
    all_df[f'arpu_vs_{col}_peers'] = all_df['arpu'] / (all_df[f'{col}_mean_arpu'] + 1e-5)
    all_df[f'{col}_mean_m1_bal'] = all_df.groupby(col)['m1_daily_avg_bal'].transform('mean')
    all_df[f'm1_bal_vs_{col}_peers'] = all_df['m1_daily_avg_bal'] / (all_df[f'{col}_mean_m1_bal'] + 1e-5)
train_fe = all_df.copy()

print("--- Applying K-Fold Target Encoding ---")
train_fe[TARGET] = y
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
for c in cat_cols: train_fe[f'{c}_te'] = np.nan
for tr_idx, val_idx in kf_te.split(train_fe):
    X_tr, X_val = train_fe.iloc[tr_idx], train_fe.iloc[val_idx]
    for c in cat_cols: train_fe.loc[val_idx, f'{c}_te'] = X_val[c].map(X_tr.groupby(c)[TARGET].mean())

INITIAL_FEATURES = [c for c in train_fe.columns if c not in [TARGET, "profile_hash"]]
print("--- Running Feature Selection ---")
temp_X = train_fe[INITIAL_FEATURES].copy()
for c in cat_cols:
    if c in temp_X.columns:
        temp_X[c] = temp_X[c].astype('category')
temp_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
temp_model.fit(temp_X, y)
best_features = pd.DataFrame({'Feature': INITIAL_FEATURES, 'Importance': temp_model.feature_importances_}).sort_values('Importance', ascending=False).head(200)['Feature'].tolist()
for c in cat_cols:
    if c not in best_features and c in INITIAL_FEATURES: best_features.append(c)

FEATURE_COLS = best_features
X = train_fe[FEATURE_COLS].values
cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in cat_cols]

for i, c in enumerate(FEATURE_COLS):
    if c in cat_cols and c in train_fe.columns:
        X[:, i] = LabelEncoder().fit_transform(train_fe[c].astype(str).values)

groups = train_fe["profile_hash"].values
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups))

def objective_lgb(trial):
    params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
            "objective": "binary"
        }
    oof_preds = np.zeros(len(X))
    for tr_idx, val_idx in splits:
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], categorical_feature=cat_indices, callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return log_loss(y, oof_preds)

def objective_xgb(trial):
    params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbosity": 0,
            "eval_metric": "logloss"
        }
    oof_preds = np.zeros(len(X))
    for tr_idx, val_idx in splits:
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return log_loss(y, oof_preds)

def objective_cb(trial):
    params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "random_seed": 42,
            "verbose": 0,
            "early_stopping_rounds": 50
        }
    oof_preds = np.zeros(len(X))
    for tr_idx, val_idx in splits:
        model = CatBoostClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], cat_features=cat_indices, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return log_loss(y, oof_preds)

print("\n-----------------------------")
print("Starting LIGHTGBM TUNING (40 Trials)")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=40)

print("\n-----------------------------")
print("Starting XGBOOST TUNING (40 Trials)")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=40)

print("\n-----------------------------")
print("Starting CATBOOST TUNING (40 Trials)")
study_cb = optuna.create_study(direction="minimize")
study_cb.optimize(objective_cb, n_trials=40)

print("\n OPTUNA TUNING COMPLETE")

print("\n BEST LIGHTGBM PARAMS")
for k, v in study_lgb.best_params.items(): print(f' "{k}": {v},')
print(f"LGB Best LogLoss: {study_lgb.best_value:.5f}")

print("\n BEST XGBOOST PARAMS")
for k, v in study_xgb.best_params.items(): print(f' "{k}": {v},')
print(f"XGB Best LogLoss: {study_xgb.best_value:.5f}")

print("\n BEST CATBOOST PARAMS")
for k, v in study_cb.best_params.items(): print(f' "{k}": {v},')
print(f"CB Best LogLoss: {study_cb.best_value:.5f}")

best_params_log = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "lightgbm": {
        "best_params": study_lgb.best_params,
        "best_logloss": study_lgb.best_value
    },
    "xgboost": {
        "best_params": study_xgb.best_params,
        "best_logloss": study_xgb.best_value
    },
    "catboost": {
        "best_params": study_cb.best_params,
        "best_logloss": study_cb.best_value
    }
}

save_path = "best_params.json"
with open(save_path, 'w') as f:
    json.dump(best_params_log, f, indent=4)

print(f"\nBest parameters saved to {save_path}")
