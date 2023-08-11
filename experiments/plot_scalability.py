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
plt.rc('text', usetex = False)
sns.set_palette("colorblind")

scl_df = pd.read_csv(SCRIPT_DIR / "../results/results-scale.csv")

scl_df["method"].replace({
        "streed_pwc_kmeans1_tasklb1_lb1_terminal0": "STreeD-CR (no d2)",
        "streed_pwc_kmeans1_tasklb1_lb1_terminal1": "STreeD-CR",
        "streed_pwl": "STreeD-LR",
        "osrt": "OSRT",
        "ort": "ORT",
        "ort-l": "ORT-L", 
        "dtip": "DTIP"
    }, inplace=True)

print(scl_df.head(5))


rts = scl_df.groupby(["train_data", "method", "depth"])["time"].mean().unstack("method")
methods = scl_df["method"].unique() 
instances_above_1s_ix = np.column_stack([rts[m] >= .1 for m in methods]).any(axis=1)
instances_above_1s = pd.Series(instances_above_1s_ix, index=rts.index)
_scl_df = scl_df[scl_df.apply(lambda x: instances_above_1s.loc[x["train_data"], x["depth"]], axis=1)]


if SPLIT_PER_CATEGORY:
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.1, 1.7), sharey=True)

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
        g.set(xscale="log", xlim=[.1, 600])
        g.set_xlabel("Runtime (s)")
        sns.move_legend(g, "lower right", bbox_to_anchor=(1, 0), ncol=1, title="", frameon=True)
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
    g.set_xlabel("Runtime (s)")

    plt.subplots_adjust(right=0.62)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.05), ncol=1, title="", frameon=True)

#plt.tight_layout()
#plt.show()
plt.savefig(SCRIPT_DIR / "plot" / "fig-scalability.pdf", bbox_inches="tight", pad_inches = 0)
