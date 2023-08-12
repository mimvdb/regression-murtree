import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

df = pd.read_csv(SCRIPT_DIR / ".." / "results" / "results-scale-dim.csv")
df = df[df["time"] < df["timeout"]]
df = df[df["time"] > 0]

df["method"].replace({"streed_pwc_kmeans1_tasklb1_lb1_terminal1": "STreeD-CR",
                      "streed_pwc_kmeans1_tasklb1_lb1_terminal0": "STreeD-CR (no d2)",
                      "osrt": "OSRT",
                      "streed_pwl": "STreeD-LR"}, inplace=True)

df_n = df[df["train_data"].str.startswith("household")].copy()
df_n["instances"] = df_n["train_data"].str.split("_", expand=True)[1].str.replace("size","").astype(int)


df_f = df[df["train_data"].str.startswith("seoul-bike")].copy()
df_f["features"] = df_f["train_data"].str.split("_", expand=True)[2].str.replace("feat","").astype(int)


sns.set_context('paper')
plt.rc('font', size=10, family='serif')
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('axes', labelsize='small', grid=True)
plt.rc('legend', fontsize='x-small')
plt.rc('pdf',fonttype = 42)
plt.rc('ps',fonttype = 42)
plt.rc('text', usetex = True)
sns.set_palette("colorblind")


fig, (ax_n, ax_f) = plt.subplots(2, 1, figsize=(3.30 + 0.2, 3))

gn = sns.lineplot(data=df_n, x="instances", y="time", ax=ax_n,
              hue="method")

gf = sns.lineplot(data=df_f, x="features", y="time", ax=ax_f,
              hue="method", legend=False)

gn.set_ylabel("Run time (s)")
gn.set_xlabel("Number of instances")
gn.set_xscale("log")
gn.set_yscale("log")
gn.set_xlim(left=100, right=1e6)

gf.set_ylabel("Run time (s)")
gf.set_xlabel("Number of features")
gf.set_yscale("log")
gf.set_xlim(left=10, right=50)

sns.move_legend(gn, "lower right", bbox_to_anchor=(1, 0), ncol=1, title="", frameon=True)

plt.tight_layout()

#plt.show()
plt.savefig(SCRIPT_DIR / "plot" / "fig-scale-dim.pdf", bbox_inches="tight", pad_inches = 0)