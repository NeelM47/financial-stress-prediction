import pandas as pd

# Load the two safe submissions you just made
sub1 = pd.read_csv("submissions/submission_seed_42.csv")
sub2 = pd.read_csv("submissions/submission_seed_999.csv")

# Blend them evenly
final_blend = sub1.copy()
final_blend['Target'] = (sub1['Target'] + sub2['Target']) / 2.0

final_blend.to_csv("submissions/submission_blend.csv", index=False)
print("Blend completed! Submit this file!")
