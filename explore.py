import re
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

OUT_DIR = "eda"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]

TX_TYPES = [
    "paybill", "merchantpay", "transfer_from_bank",
    "mm_send", "received", "deposit", "withdraw",
]
TX_ENTITIES = {
    "paybill": "companies", "merchantpay": "merchants",
    "transfer_from_bank": "banks", "mm_send": "recipients",
    "received": "senders", "deposit": "agents", "withdraw": "agents",
}

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")
sub = pd.read_csv("data/SampleSubmission.csv")
y = train[TARGET]

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

section("1. HIGH-LEVEL STRUCTURE")
print(f"Train shape: {train.shape} | Test shape: {test.shape} | SampleSubmission shape: {sub.shape}")
print(f"Train memory: {train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Unique IDs in Train: {train[ID_COL].nunique()} | Test: {test[ID_COL].nunique()}")
print(f"Duplicate IDs: Train={train[ID_COL].duplicated().sum()}, Test={test[ID_COL].duplicated().sum()}")

n_num = train.select_dtypes("number").shape[1]
n_cat = train.select_dtypes("object").shape[1]
print(f"Numeric columns: {n_num} | Categorical/object columns: {n_cat}")

section("2. TARGET VARIABLE")
print("Label meanings: 0 = no liquidity stress, 1 = liquidity stress next 30 days")
vc = train[TARGET].value_counts()
print(vc.to_string())
print(f"\nPositive rate: {vc[1] / len(train):.3f} ({vc[1]} rows)  - imbalanced (~85/15)")

section("3. MISSING VALUES")
miss_train = train.isna().sum()
miss_cols = miss_train[miss_train > 0].sort_values(ascending=False)
if len(miss_cols):
    print(f"{len(miss_cols)} columns have missing values in Train:")
    print(pd.DataFrame({"n_missing": miss_cols, "pct": (miss_cols / len(train)).round(4)}).head(20).to_string())
else:
    print("No missing values in Train.")
print(f"Missing values in test: {test.isna().sum().sum()}")

section("4. CATEGORICAL COLUMNS")
for c in CAT_COLS:
    print(f"\n--- {c} (unique: {train[c].nunique()}) ---")
    counts = train[c].value_counts(dropna=False)
    t = pd.DataFrame({"count": counts, "pct": (counts / len(train)).round(4)})
    t["stress_rate"] = train.groupby(c)[TARGET].mean().reindex(t.index).round(3)
    print(t.head(15).to_string())

section("5. UNIQUE VALUES IN NUMERIC COLUMNS")
num_cols = [c for c in train.columns if train[c].dtype != object and c not in [TARGET, ID_COL]]
zeros = (train[num_cols] == 0).sum().sort_values(ascending=False)
print("Columns with the most zeros (dormancy indicators):")
print(zeros.head(10).to_string())

section("6. MONTH STRUCTURE (M1 = most recent ... M6 = oldest)")
m_pattern = re.compile(r"^m([1-6])_([a-z_]+)_(volume|total_value|highest_amount|daily_avg_bal)$|^m([1-6])_([a-z_]+)_([a-z_]+)$")
months_of_cols = {}
for c in train.columns:
    m = re.match(r"^m([1-6])_", c)
    if m:
        months_of_cols.setdefault(int(m.group(1)), []).append(c)
for mo, cols in sorted(months_of_cols.items()):
    print(f"M{mo}: {len(cols)} columns (e.g. {cols[:3]})")

section("7. TRANSACTION TYPE BREAKDOWN (total value, all months combined)")
type_cols = {}
for c in train.columns:
    m = re.match(r"^m[1-6]_(paybill|merchantpay|transfer_from_bank|mm_send|received|deposit|withdraw)_(total_value|volume|highest_amount)$", c)
    if m:
        type_cols.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(c)
rows = []
for t, metrics in type_cols.items():
    rows.append({
        "txn_type": t,
        "counterparty": TX_ENTITIES.get(t),
        "total_value_sum": train[metrics["total_value"]].sum().sum(),
        "volume_sum": train[metrics["volume"]].sum().sum(),
        "highest_amt_mean": train[metrics["highest_amount"]].mean().mean(),
    })
print(pd.DataFrame(rows).sort_values("total_value_sum", ascending=False).to_string(index=False))

section("\n8. MONTH-OVER-MONTH TREND (avg daily balance)")
bal_cols = [f"m{m}_daily_avg_bal" for m in range(1, 7)]
if all(c in train.columns for c in bal_cols):
    means = train[bal_cols].mean()
    stressed = train[train[TARGET] == 1][bal_cols].mean()
    pts = pd.DataFrame({"all": means, "stressed": stressed})
    print("Average daily balance by month:")
    print(pts.to_string())
    ax = pts.plot(marker="o", title="Avg daily balance by month (M1 most recent)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/balance_by_month.png", dpi=110); plt.close()

section("\n9. FEATURES CORRELATED WITH TARGET")
feat_cols = [c for c in train.columns if c not in [TARGET, ID_COL] and pd.api.types.is_numeric_dtype(train[c])]
corrs = train[feat_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
corrs = corrs[corrs > 0].sort_values(ascending=False)
top = pd.DataFrame({"corr": corrs}).head(20)
print("Top 20 most correlated features (|r| with target):")
print(top.to_string())

section("\n10. TARGET RATE BY SEGMENT & EARNING PATTERN")
rate = train.groupby(["segment", "earning_pattern"])[TARGET].agg(["mean", "count"]).round(3)
rate = rate[rate["count"] > 100].sort_values("mean", ascending=False)
print(rate.head(20).to_string())

section("\n9. DISTRIBUTION OF KEY FEATURES BY TARGET")
for c in ["arpu", "m1_daily_avg_bal", "x_90_d_activity_rate"]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, tval, lbl in zip(axes, [0, 1], ["no stress (0)", "stressed (1)"]):
        subd = train[train[TARGET] == tval][c].clip(lower=train[c].quantile(0.01), upper=train[c].quantile(0.99))
        subd.hist(bins=60, ax=ax, alpha=0.7)
        ax.set_title(f"{c} | {lbl}")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/dist_{c}.png", dpi=110); plt.close()
print(f"Saved distribution plots to {OUT_DIR}/")

section("12. HIGHEST-AMOUNT / ENTITY DIVERSITY — PAYBILL VS MERCHANT")
pay_bill = train[[f"m1_paybill_total_value", f"m1_merchantpay_total_value"]]
print("Mean monthly value (M1):")
print(pay_bill.describe().T[['mean', '50%']].to_string())

print("\nDone. Plots saved in", OUT_DIR)
