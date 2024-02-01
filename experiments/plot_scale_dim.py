import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

df = pd.read_csv(SCRIPT_DIR / ".." / "results" / "results-scale-dim-d4.csv")
df = df[df["time"] < df["timeout"]]
df = df[df["time"] > 0]

df["d2"] = "No d2"
df.loc[(df["method"] == "streed_pwc_kmeans1_tasklb1_lb1_terminal1") | (df["method"] == "streed_pwsl_terminal1"), "d2"] = "d2"

df["method"].replace({"streed_pwc_kmeans1_tasklb1_lb1_terminal1": "SRT-C",
                      "streed_pwc_kmeans1_tasklb1_lb1_terminal0": "SRT-C",
                      "osrt": "OSRT",
                      "streed_pwl": "SRT-L",
                      "streed_pwsl_terminal1": "SRT-SL",
                      "streed_pwsl_terminal0": "SRT-SL",
                      }
                      , inplace=True)

df["method-d2"] = df["method"] + df["d2"]

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


fig, (ax_n, ax_f) = plt.subplots(2, 1, figsize=(3.30 + 0.2, 3.6))

method_order = ["SRT-C", "SRT-SL", "SRT-L", "OSRT"]
style_order = ["No d2", "d2"]
markers = {
    "SRT-CNo d2": "s", 
    "SRT-Cd2": "s", 
    "SRT-SLNo d2": "X",
    "SRT-SLd2": "X",
    "SRT-LNo d2": "o",
    "OSRTNo d2": "v"
}
gn = sns.lineplot(data=df_n, x="instances", y="time", ax=ax_n,
              hue="method", hue_order=method_order,
              style="d2", style_order=style_order)
mean_df_n = df_n.groupby(["method", "d2", "method-d2", "instances"], as_index=False, sort=False).agg({"time": "mean"})
sns.scatterplot(data=mean_df_n, x='instances', y='time', ax=ax_n, hue_order=method_order,
             hue="method", style="method-d2", markers=markers, s=25)



gf = sns.lineplot(data=df_f, x="features", y="time", ax=ax_f,
              hue="method",  hue_order=method_order,
              style="d2", style_order=style_order,
              legend=False)
mean_df_f = df_f.groupby(["method", "d2", "method-d2", "features"], as_index=False, sort=False).agg({"time": "mean"})
sns.scatterplot(data=mean_df_f, x='features', y='time', ax=ax_f, hue_order=method_order,
             hue="method", style="method-d2", markers=markers, s=25,
             legend=False)

gn.set_ylabel("Run time (s)")
gn.set_xlabel("Number of instances")
gn.set_xscale("log")
gn.set_yscale("log")
gn.set_xlim(left=100, right=1e6)
gn.set_ylim(0.01, 1000)

gf.set_ylabel("Run time (s)")
gf.set_xlabel("Number of features")
gf.set_yscale("log")
gf.set_xlim(left=10, right=50)
gf.set_ylim(0.01, 1000)

handles, labels = gn.get_legend_handles_labels()
handles[1].set_marker("s")
handles[2].set_marker("X")
handles[3].set_marker("o")
handles[4].set_marker("v")
ggz=gn.legend(handles=handles[1:5] + handles[6:8], labels=labels[1:5] + labels[6:8], loc="lower left",  bbox_to_anchor=(0, 1), ncol=3, title="", frameon=True)

plt.subplots_adjust(hspace=0, wspace=0)
plt.tight_layout()

#plt.show()
plt.savefig(SCRIPT_DIR / "plot" / "fig-scale-dim.pdf", bbox_inches="tight", pad_inches = 0)