import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = Path(__file__).parent.resolve()

WIDTH = 3.3249
HEIGTH = 2

df = pd.read_csv(SCRIPT_DIR / "../results/results-inc-node.csv")
df["dataset"] = df["train_data"].str.rsplit("_", n=2, expand=True)[0]
df = df[(df["dataset"]=='airfoil') | (df["dataset"]=='real-estate')]
n_cols = len(df["dataset"].unique())
n_methods = len(df["method"].unique())
df["dataset"] = df["dataset"].replace({"airfoil": "Airfoil", "real-estate": "Real Estate Evaluation"})
df["method"] = df["method"].replace({"cart": "CART", "iai": "IAI", "guide": "GUIDE", "streed_pwc": "STreeD"})

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


rel = sns.relplot(
    data=df, x="depth", y="test_r2",
    col="dataset", hue="method", style="method",
    kind="line",
    height = HEIGTH, aspect=(WIDTH / (n_cols*1.16))/HEIGTH,
    facet_kws={'sharey': True}
)

for ax in rel.fig.axes:
    #y_low, y_high = ax.get_ylim()
    #y_low = y_high * 0.6
    ax.set_ylim(0.4, 0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

rel.set_xlabels("Maximum depth")
rel.set_ylabels("$R^2$ score")
rel.set_titles("{col_name}")

sns.move_legend(rel, "upper center", bbox_to_anchor=(0.55, 0.87), ncol=n_methods, title="", frameon=True)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "plot" / "fig-inc-nodes.pdf", bbox_inches="tight", pad_inches = 0.03)