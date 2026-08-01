import pandas as pd

tree_sub = pd.read_csv("submissions/submission_stacked.csv")
dl_sub = pd.read_csv("submissions/submission_dl.csv")

final_blend = tree_sub.copy()

final_blend['Target'] = (tree_sub['Target'] * 0.9) + (dl_sub['Target'] * 0.1)

final_blend.to_csv("submission_TREE_DL_BLEND.csv", index=False)
