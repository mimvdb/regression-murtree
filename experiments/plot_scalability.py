import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import gmean

SCRIPT_DIR = Path(__file__).parent.resolve()

SPLIT_PER_CATEGORY = True
SPLIT_PER_DEPTH    = True

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


files = [
    "d1-results.csv",
    #"d2-predetermined_timeouts.csv"
    "d2-mip-results.csv",
    "d2-results.csv",
    "d3-mip-results.csv",
    "d3-results.csv",
    "d4-mip-results.csv",
    #"d4-predetermined_timeouts.csv",
    "d4-results.csv",
    "d5-mip-results.csv",
    "d5-predetermined_timeouts.csv",
    "d5-results.csv",
    "d6-mip-results.csv",
    "d6-predetermined_timeouts.csv",
    "d6-results.csv",
    "d7-mip-results.csv",
    "d7-predetermined_timeouts.csv",
    "d7-results.csv",
    "d8-mip-results.csv",
    "d8-predetermined_timeouts.csv",
    "d8-results.csv",
    "d9-mip-results.csv",
    "d9-predetermined_timeouts.csv",
    "d9-results.csv",
]

dfs = [pd.read_csv(SCRIPT_DIR / f"../results/scale_results/{file}") for file in files]
scl_df = pd.concat(dfs)

scl_df["method"] = scl_df["method"].replace({
        "streed_pwc_kmeans1_tasklb1_lb1_terminal0": "SRT-C (no d2)",
        "streed_pwc_kmeans1_tasklb1_lb1_terminal1": "SRT-C",
        "streed_pwsl_terminal0": "SRT-SL (no d2)",
        "streed_pwsl_terminal1": "SRT-SL",
        "streed_pwl": "SRT-L",
        "osrt": "OSRT",
        "ort_lFalse_metricMAE": "ORT",
        "ort_lTrue_metricMAE": "ORT-L", 
        "dtip": "DTIP"
    })

scl_df.loc[scl_df["time"] == -1, "time"] = scl_df.loc[scl_df["time"] == -1, "timeout"] * 2
scl_df.loc[scl_df["time"] == scl_df["timeout"] + 1, "time"] = scl_df.loc[scl_df["time"] == scl_df["timeout"] + 1, "timeout"] * 2

rts = scl_df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")


#scl_df.groupby(["train_data", "method", "depth"])["time"].mean().to_csv("out.csv")

methods = scl_df["method"].unique() 
instances_above_1s_ix = np.column_stack([rts[m] >= 1 for m in methods]).any(axis=1)
instances_above_1s = pd.Series(instances_above_1s_ix, index=rts.index)
_scl_df = scl_df#[scl_df.apply(lambda x: instances_above_1s.loc[x["train_data"], x["depth"]], axis=1)]

if SPLIT_PER_CATEGORY:

    if SPLIT_PER_DEPTH:
        fig, (axs1, axs2, axs3, axs4) = plt.subplots(4, 3, figsize=(7.9, 5.0), sharey=True, sharex=True)
        row1 = (axs1, [1,2])
        row2 = (axs2, [3,4])
        row3 = (axs3, [5,6])
        row4 = (axs4, [7,8,9])
        rows = [row1, row2, row3, row4]
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.9, 1.3), sharey=True)
        row1 = ((ax1, ax2, ax3), list(range(1,10)))
        rows = [row1]
    
    colors = sns.color_palette("colorblind", 7)
    
    first_row = True
    for axs, depths in rows:
        ax1, ax2, ax3 = axs
        _scl_df_ = _scl_df[_scl_df["depth"].isin(depths)]

        method_order = ["SRT-C", "OSRT", "ORT", "DTIP"]
        g1 = sns.ecdfplot(data=_scl_df_, x="time", ax=ax1,
                    stat="proportion", hue='method', hue_order=method_order
            )
        g1_style = [
            (colors[0], "-", "s", 3),
            (colors[3], "-", "o", 3),
            (colors[5], "-", "v", 3),
            (colors[4], "-", "D", 3),
        ]

        method_order = ["SRT-SL", "SRT-L", "ORT-L"]
        g2 = sns.ecdfplot(data=_scl_df_, x="time", ax=ax2,
                    stat="proportion", hue='method', hue_order=method_order
            )
        g2_style = [
            (colors[1], "-", "X", 4),
            (colors[2], "-", "^", 3),
            (colors[6], "-", "<", 3),
        ]

        method_order = [ "SRT-C", "SRT-C (no d2)", "SRT-SL", "SRT-SL (no d2)"]
        g3 = sns.ecdfplot(data=_scl_df_, x="time", ax=ax3,
                    stat="proportion", hue='method', hue_order=method_order
            )
        g3_style = [
            (colors[0], "-", "s", 3),
            (colors[0], "--", "s", 3),
            (colors[1], "-", "X", 4),
            (colors[1], "--", "X", 4),
        ]
        
        if SPLIT_PER_DEPTH:
            low = min(depths)
            high = max(depths)
            g1.set_ylabel(f"Depth = {low}-{high}\nTrees computed")    
        else:
            g1.set_ylabel("Trees computed")    
        g1.yaxis.set_major_formatter(PercentFormatter(1))
        if first_row:
            g1.set_title("Piecewise constant methods")
            g2.set_title("Piecewise linear methods")
            g3.set_title("Effect of depth-two solver")

        for g, g_styles, legend_position, legend_anchor, in [(g1, g1_style, "lower left", (0,0)), (g2, g2_style, "lower left", (0,0)), (g3, g3_style, "lower right", (1,0))]:
            if SPLIT_PER_DEPTH:
                g.set(xscale="log", xlim=[.01, 1000])
            else:
                g.set(xscale="log", xlim=[1, 1000])
            
            g.set_xlabel("Run time (s)")
            
            if first_row:
                sns.move_legend(g, legend_position, bbox_to_anchor=legend_anchor, ncol=1, title="", frameon=True)
                _handles = g.legend_.legendHandles
                #print(g.lines, g_styles, _handles)

                for lines, linestyle, legend_handle in zip(g.lines[::-1], g_styles, _handles):
                    #print(linestyle)
                    lines.set_color(linestyle[0])
                    lines.set_linestyle(linestyle[1])
                    lines.set_marker(linestyle[2])
                    lines.set_markevery(0.2)
                    lines.set_markersize(linestyle[3])
                    legend_handle.set_color(linestyle[0])
                    legend_handle.set_linestyle(linestyle[1])
                    legend_handle.set_marker(linestyle[2])
                    legend_handle.set_markersize(linestyle[3])

            else:
                g.get_legend().remove()

                for lines, linestyle in zip(g.lines[::-1], g_styles):
                    #print(linestyle)
                    lines.set_color(linestyle[0])
                    lines.set_linestyle(linestyle[1])
                    lines.set_marker(linestyle[2])
                    lines.set_markevery(0.2)
                    lines.set_markersize(linestyle[3])
        
        first_row = False

    plt.subplots_adjust(wspace=0.12)

else:

    plt.figure(figsize=(3.25, 1.85))

    method_order = ["DTIP", "ORT", "ORT-L", "OSRT", "SRT-C", "SRT-C (no d2)", "STreeD-LR"]
    g = sns.ecdfplot(data=_scl_df, x="time",
                stat="proportion", hue='method', hue_order=method_order)

    g.legend_.set_title("")

    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.'][:len(g.legend_.legend_handles)]
    for lines, linestyle, legend_handle in zip(g.lines[::-1], line_styles, g.legend_.legend_handles[-3:]):
        lines.set_linestyle(linestyle)
        legend_handle.set_linestyle(linestyle)

    g.set_ylabel("Trees computed")    
    g.yaxis.set_major_formatter(PercentFormatter(1))

    g.set(xscale="log", xlim=[.1, 600])
    g.set_xlabel("Run time (s)")

    plt.subplots_adjust(right=0.62)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.05), ncol=1, title="", frameon=True)

#plt.tight_layout()
#plt.show()
filename = "fig-scalability.pdf"
if SPLIT_PER_DEPTH: filename = "fig-depth-split-scalability.pdf"
plt.savefig(SCRIPT_DIR / "plot" / filename, bbox_inches="tight", pad_inches = 0)

rts = _scl_df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")
instances_within_time_out_ix = np.column_stack([rts[m] < 1000 for m in ["OSRT", "SRT-C"]]).all(axis=1)
instances_within_time_out = pd.Series(instances_within_time_out_ix, index=rts.index)
_scl_df = _scl_df[_scl_df.apply(lambda x: instances_within_time_out.loc[x["train_data"], x["depth"]], axis=1)]

r = _scl_df.groupby(["train_data", "depth", "method"])["time"].mean().unstack("method")
print((r["OSRT"] / r["SRT-C"]).unstack("depth"))
rel_perf = gmean((r["OSRT"] / r["SRT-C"]).unstack("depth"), axis=0, nan_policy="omit")
print("OSRT vs SRT-C", rel_perf)
rel_perf = gmean(r["OSRT"] / r["SRT-C"])
print("OSRT vs SRT-C", rel_perf)

rel_perf = gmean((r["SRT-C (no d2)"] / r["SRT-C"]).unstack("depth"), axis=0, nan_policy="omit")
print("C | No d2 vs with d2", rel_perf)
rel_perf = gmean(r["SRT-C (no d2)"] / r["SRT-C"])
print("C | No d2 vs with d2", rel_perf)

rel_perf = gmean((r["SRT-SL (no d2)"] / r["SRT-SL"]).unstack("depth"), axis=0, nan_policy="omit")
print("SL | No d2 vs with d2", rel_perf)
rel_perf = gmean(r["SRT-SL (no d2)"] / r["SRT-SL"])
print("SL | No d2 vs with d2", rel_perf)