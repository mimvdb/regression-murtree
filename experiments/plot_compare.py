import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

OUTPUT_TEX_TABLE = True

df = pd.read_csv(SCRIPT_DIR / "../results/results-tune.csv")
df["dataset"] = df["train_data"].str.rsplit("_", n=2, expand=True)[0]

df = df[df["dataset"] != "household"]

df.loc[((df["method"] == "iai") | (df["method"] == "iai_l")) & (df["time"] > df["timeout"] * 2), "test_r2"] = -1 # * 2 because leniency for methods run locally

#df = df[(df["method"] != "cart") & (df["method"] != "iai") & (df["method"] != "streed_pwc") & (df["method"] != "guide") & (df["method"] != "lr")  & (df["method"] != "osrt")]
#df = df[(df["method"] != "iai_l") & (df["method"] != "streed_pwl") & (df["method"] != "guide_l") & (df["method"] != "lr")]

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
        if max(abs(diff)) <= 1e-6:
               print(f"{m1} = {m2}, m = {med}, p = 1.0")
        else:
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


if OUTPUT_TEX_TABLE:
    print("")
    method_order = ["LR", "CART", "GUIDE", "IAI", "OSRT", "SRT-C", "GUIDE-SL", "SRT-SL", "GUIDE-L", "IAI-L", "SRT-L (Lasso)", "SRT-L (Elastic Net)"]
    method_map = {
        "LR": "lr",
        "CART": "cart",
        "GUIDE": "guide",
        "GUIDE-SL": "guide_l",
        "GUIDE-L": "guide_l",
        "OSRT": "osrt",
        "SRT-C": "streed_pwc",
        "SRT-SL": "streed_pwsl",
        "SRT-L (Elastic Net)": "streed_pwl_elasticnet",
        "SRT-L (Lasso)": "streed_pwl_lasso",
        "IAI": "iai",
        "IAI-L": "iai_l"
    }

    print("Best per category &")
    
    for method in method_order:
        sep = "\\\\" if method == method_order[-1] else "&"
        if method == "LR":
            print(f"{sep} % {method}")
            continue
        if method in ["CART", "GUIDE", "IAI", "OSRT", "SRT-C"]:
            df2 = df[(df["method"] != "iai_l") & (df["method"] != "streed_pwl") & (df["method"] != "guide_l") & (df["method"] != "lr")]
        elif method in ["SRT-SL", "GUIDE-SL"]:
            df2 = df[(df["method"] != "guide-sl") & (df["method"] != "streed_pwsl") ]
        else:
            df2 = df[(df["method"] != "cart") & (df["method"] != "iai") & (df["method"] != "streed_pwc") & (df["method"] != "guide") & (df["method"] != "lr") & (df["method"] != "osrt")]

        means = df2.groupby(["dataset", "method"])["test_r2"].mean()
        means = means.unstack('method')

        m = method_map[method]
        if m not in means:
            print(f"? {sep} % {method}")
            continue
        best_scores = means.max(axis=1)
        scores = means[m]
        wins = sum(round(scores, 2) >= round(best_scores, 2))
        print(f"{wins} {sep} % {method}")

    print("Best overall &")
    for method in method_order:
        sep = "\\\\" if method == method_order[-1] else "&"

        means = df.groupby(["dataset", "method"])["test_r2"].mean()
        means = means.unstack('method')

        m = method_map[method]
        if m not in means:
            print(f"? {sep} % {method}")
            continue
        best_scores = means.max(axis=1)
        scores = means[m]
        wins = sum(round(scores, 2) >= round(best_scores, 2))
        print(f"{wins} {sep} % {method}")

