import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

df = pd.read_csv(SCRIPT_DIR / "results.csv")

methods = df["method"].unique()

for i, m1 in enumerate(methods):
    for j, m2 in enumerate(methods[i+1:]):
        
        m1_r2 = df[df["method"] == m1]["test_r2"].reset_index(drop=True)
        m2_r2 = df[df["method"] == m2]["test_r2"].reset_index(drop=True)

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