import pandas as pd

print("Loading data ...")
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

preds = pd.read_csv("submissions/submission_blend.csv")

test_with_preds = test.copy()
test_with_preds['Target'] = preds['Target']

LOWER_THRESH = 0.015
UPPER_THRESH = 0.90

pseudo_0 = test_with_preds[test_with_preds['Target'] < LOWER_THRESH].copy()
pseudo_0['liquidity_stress_next_30d'] = 0

pseudo_1 = test_with_preds[test_with_preds['Target'] > UPPER_THRESH].copy()
pseudo_1['liquidity_stress_next_30d'] = 1

pseudo_df = pd.concat([pseudo_0, pseudo_1], axis=0).drop(columns=['Target'])

print(f"Found {len(pseudo_0)} highly confident 0s.")
print(f"Found {len(pseudo_1)} highly confident 1s.")
print(f"Adding {len(pseudo_df)} new rows to the training data...")


new_train = pd.concat([train, pseudo_df], axis=0).reset_index(drop=True)

new_train.to_csv("data/Train_Pseudo.csv", index=False)
print("\nSuccess saved as data/Train_Pseudo.csv")
