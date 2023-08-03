import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

df = pd.read_csv(SCRIPT_DIR / "results.csv")
df["dataset"] = df["train_data"].str.rsplit("_", n=1, expand=True)[0]

print("\n * Mean Test R2 * \n")
methods = df["method"].unique()
means = df.groupby(["dataset", "method"])["test_r2"].mean()
means = means.unstack('method')
print(means)

print("\n * Number of wins (folds): *\n")
best_scores = df.groupby(["train_data"])["test_r2"].max()
for i, m in enumerate(methods):
    scores = df[df["method"] == m]
    scores.index = scores["train_data"]
    scores = scores["test_r2"]
    wins = sum(scores >= best_scores)
    print(f"{m}: {wins} wins")

print("\n * Number of wins (mean per dataset): *\n")
best_scores = means.max(axis=1)
for i, m in enumerate(methods):
    scores = means[m]
    wins = sum(scores + 1e-3 >= best_scores)
    print(f"{m}: {wins} wins")

print("\n * Wilcoxon comparison: *\n")
for i, m1 in enumerate(methods):
    for j, m2 in enumerate(methods[i+1:]):
        
        m1_r2 = df[df["method"] == m1]["test_r2"].reset_index(drop=True)
        m2_r2 = df[df["method"] == m2]["test_r2"].reset_index(drop=True)

        no_timeouts = (m1_r2 != -1) & (m2_r2 != -1)
        m1_r2 = m1_r2[no_timeouts]
        m2_r2 = m2_r2[no_timeouts]

        diff = m1_r2 - m2_r2
        med = diff.median()
        _, p = wilcoxon(diff)
        if p >=  0.05: 
            print(f"{m1} = {m2}, m = {med}, p = {p}")
        else:
            _, p_less = wilcoxon(diff, alternative='less')
            _, p_greater = wilcoxon(diff, alternative='greater')
            if p_less < p_greater:
                print(f"{m1} < {m2}, m = {med}, p = {p}")
            else:
                print(f"{m1} > {m2}, m = {med}, p = {p}")