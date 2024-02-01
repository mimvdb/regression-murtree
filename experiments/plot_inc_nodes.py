import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = Path(__file__).parent.resolve()

WIDTH = 3.3249 + 0.1
HEIGTH = 1.7

df = pd.read_csv(SCRIPT_DIR / "../results/results-inc-node.csv")
df["dataset"] = df["train_data"].str.rsplit("_", n=2, expand=True)[0]
df = df[(df["dataset"]=='airfoil') | (df["dataset"]=='real-estate')]
n_cols = len(df["dataset"].unique())
n_methods = len(df["method"].unique())
df["dataset"] = df["dataset"].replace({"airfoil": "Airfoil", "real-estate": "Real Estate Evaluation"})
df["method"] = df["method"].replace({"cart": "CART", "iai": "IAI", "guide": "GUIDE", 
                                     "streed_pwc_kmeans1_tasklb1_lb1_terminal1": "SRT-C",
                                     "cart-bin": "CART (Binary)",
                                     "iai-bin": "IAI (Binary)"})
df["method_org"] = df["method"].str.replace(" (Binary)", "", regex=False)

df["binary"] = df["method"].apply(lambda x: "Binary" if x.endswith("(Binary)") else "Original")
df.loc[df["method"] == "SRT-C", "binary"] = "Binary"

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
    col="dataset", hue="method_org", style="binary",
    hue_order=["SRT-C", "IAI", "CART"],
    style_order=["Binary", "Original"],
    kind="line",
    height = HEIGTH, aspect=WIDTH / HEIGTH ,
    facet_kws={'sharey': True}, 
    legend=False
)

for ax in rel.fig.axes:
    ax.set_ylim(bottom=0.4)
    ax.set_xlim(2,7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(ax.lines, ["SRT-C", "IAI (Binary)", "IAI", "CART (Binary)", "CART"])

    

rel.set_xlabels("Maximum depth")
rel.set_ylabels("$R^2$ score (Train)")
rel.set_titles("")


plt.tight_layout()
plt.savefig(SCRIPT_DIR / "plot" / "fig-inc-nodes.pdf", bbox_inches="tight", pad_inches = 0.03)