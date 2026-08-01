import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("\n--- Running Adversarial Validation ---")
train_adv = train_fe[FEATURE_COLS].copy()
train_adv['is_test'] = 0

test_adv = test_fe[FEATURE_COLS].copy()
test_adv['is_test'] = 1

adv_data = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
X_adv = adv_data.drop('is_test', axis=1)
y_adv = adv_data['is_test']

for c in cat_cols:
    if c in X_adv.columns:
        X_adv[c] = X_adv[c].astype('category')

X_train, X_val, y_train, y_val = train_test_split(X_adv, y_adv, test_size=0.33, random_state=42)

adv_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
adv_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])

adv_auc = roc_auc_score(y_val, adv_model.predict_proba(X_val)[:, 1])
print(f"\nAdversarial ROC-AUC: {adv_auc:.4f}")


if adv_auc > 0.60:
    print("WARNING: Train and Test distributions are different (Drift detected)!")
    importance = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': adv_model.feature_importances_})
    print("These features are causing the drift (Consider deleting the top ones):")
    print(importance.sort_values('Importance', ascending=False).head(15))
else:
    print("SUCCESS: Train and Test distributions look identical. No drift detected.")
