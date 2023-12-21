import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()

SPLIT_PER_CATEGORY = True

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

# files = ["results-after-fix-streed-d1-5.csv",
#          "results-d1-p1.csv",
#          "results-d1-p2.csv",
#          "results-d1-p3.csv",
#          "results-d2-p1.csv",
#          "results-d2-p2.csv",
#          "results-d2-p3.csv",
#          "results-d3-p1.csv",
#          "results-d3-p2.csv",
#          "results-d4.csv",
#          "results-d5.csv",
#          "results-d6.csv",
#          "timeouts-d2.csv",
#          "timeouts-d3.csv",
#          "timeouts-d4.csv",
#          "timeouts-d5.csv",
#          "timeouts-d6.csv"]

# dfs = [pd.read_csv(SCRIPT_DIR / f"../results/scalability/{file}") for file in files]
# scl_df = pd.concat(dfs)
scl_df = pd.read_csv(SCRIPT_DIR / "../results/results-scale.csv")

scl_df["method"].replace({
        "streed_pwc_kmeans1_tasklb1_lb1_terminal0": "STreeD-CR (no d2)",
        "streed_pwc_kmeans1_tasklb1_lb1_terminal1": "STreeD-CR",
        "streed_pwl": "STreeD-LR",
        "osrt": "OSRT",
        "ort_lFalse_metricMAE": "ORT",
        "ort_lTrue_metricMAE": "ORT-L", 
        "dtip": "DTIP"
    }, inplace=True)

scl_df.loc[scl_df["time"] == -1, "time"] = scl_df.loc[scl_df["time"] == -1, "timeout"] * 2
scl_df.loc[scl_df["time"] == scl_df["timeout"] + 1, "time"] = scl_df.loc[scl_df["time"] == scl_df["timeout"] + 1, "timeout"] * 2

print(scl_df.head(5))


rts = scl_df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")
methods = scl_df["method"].unique() 
instances_above_1s_ix = np.column_stack([rts[m] >= 1 for m in methods]).any(axis=1)
instances_above_1s = pd.Series(instances_above_1s_ix, index=rts.index)
_scl_df = scl_df[scl_df.apply(lambda x: instances_above_1s.loc[x["train_data"], x["depth"]], axis=1)]


if SPLIT_PER_CATEGORY:
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.1, 1.3), sharey=True)

    method_order = ["STreeD-CR", "OSRT", "ORT", "DTIP"]
    g1 = sns.ecdfplot(data=_scl_df, x="time", ax=ax1,
                stat="proportion", hue='method', hue_order=method_order
        )
    method_order = ["STreeD-LR", "ORT-L"]
    g2 = sns.ecdfplot(data=_scl_df, x="time", ax=ax2,
                stat="proportion", hue='method', hue_order=method_order
        )
    method_order = [ "STreeD-CR", "STreeD-CR (no d2)"]
    g3 = sns.ecdfplot(data=_scl_df, x="time", ax=ax3,
                stat="proportion", hue='method', hue_order=method_order
        )
    
    g1.set_ylabel("Trees computed")    
    g1.yaxis.set_major_formatter(PercentFormatter(1))
    g1.set_title("Piecewise constant methods")
    g2.set_title("Piecewise linear methods")
    g3.set_title("Effect of depth-two solver")

    for g in [g1, g2, g3]:
        g.set(xscale="log", xlim=[1, 1000])
        g.set_xlabel("Run time (s)")
        sns.move_legend(g, "lower left", bbox_to_anchor=(0, 0), ncol=1, title="", frameon=True)
        line_styles = ['-', '--', ':', '-.']
        for lines, linestyle, legend_handle in zip(g.lines[::-1], line_styles, g.legend_.legend_handles):
            lines.set_linestyle(linestyle)
            legend_handle.set_linestyle(linestyle)

    plt.subplots_adjust(wspace=0.1)

else:

    plt.figure(figsize=(3.25, 1.85))

    method_order = ["DTIP", "ORT", "ORT-L", "OSRT", "STreeD-CR", "STreeD-CR (no d2)", "STreeD-LR"]
    g = sns.ecdfplot(data=_scl_df, x="time",
                stat="proportion", hue='method', hue_order=method_order)

    g.legend_.set_title("")

    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.'][:len(g.legend_.legend_handles)]
    for lines, linestyle, legend_handle in zip(g.lines[::-1], line_styles, g.legend_.legend_handles[-3:]):
        lines.set_linestyle(linestyle)
        legend_handle.set_linestyle(linestyle)

    g.set_ylabel("Trees computed")    
    g.yaxis.set_major_formatter(PercentFormatter(1))

    #g.set(xscale="log", xlim=[1e-3, 600])
    g.set(xscale="log", xlim=[.1, 600])
    g.set_xlabel("Run time (s)")

    plt.subplots_adjust(right=0.62)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.05), ncol=1, title="", frameon=True)

#plt.tight_layout()
#plt.show()
plt.savefig(SCRIPT_DIR / "plot" / "fig-scalability.pdf", bbox_inches="tight", pad_inches = 0)


rts = _scl_df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")
instances_within_time_out_ix = np.column_stack([rts[m] < 1000 for m in ["OSRT", "STreeD-CR"]]).all(axis=1)
instances_within_time_out = pd.Series(instances_within_time_out_ix, index=rts.index)
_scl_df = _scl_df[_scl_df.apply(lambda x: instances_within_time_out.loc[x["train_data"], x["depth"]], axis=1)]

r = _scl_df.groupby(["train_data", "depth", "method"])["time"].mean().unstack("method")
rel_perf = (r["OSRT"] / r["STreeD-CR"]).unstack("depth").mean()
print("OSRT vs STreeD-CR", rel_perf)
rel_perf = (r["OSRT"] / r["STreeD-CR"]).mean()
print("OSRT vs STreeD-CR", rel_perf)

rel_perf = (r["STreeD-CR (no d2)"] / r["STreeD-CR"]).unstack("depth").mean()
print("No d2 vs with d2", rel_perf)
rel_perf = (r["STreeD-CR (no d2)"] / r["STreeD-CR"]).mean()
print("No d2 vs with d2", rel_perf)