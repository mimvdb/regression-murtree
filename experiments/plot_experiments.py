from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import os

algs = ["streed_none", "streed_all", "streed_equivalent", "streed_similarity", "osrt"]
alg_name = {
    "streed_none": "STreeD (None)",
    "streed_all": "STreeD (k-Means Equiv. Points + Similarity)",
    "streed_equivalent": "STreeD (Equivalent Points + Similarity)",
    "streed_similarity": "STreeD (Similarity)",
    "osrt": "OSRT"
}
fig_formats = [".svg", ".png"]

# cost_complexity_bins = [0, 0.003, 0.01, 0.5]

def read_all(datasets, algs):
    frames = []
    for dataset in datasets:
        print(f"Preparing data for {dataset}")
        dframes = []
        for alg in algs:
            print(f"Reading data for {alg}")
            alg_frame = pd.read_csv(f"./results/{dataset}/{alg}.csv")
            alg_frame["Algorithm"] = alg_name[alg]
            alg_frame["Dataset"] = dataset
            # alg_frame["Complexity cost"] = pd.cut(alg_frame["cost_complexity"], cost_complexity_bins)
            # cost_categories = alg_frame["Complexity cost"].cat.categories
            frames.append(alg_frame)
            dframes.append(alg_frame)
            # np.abs to get rid of warning for negatives. They will get filtered out anyway by the drop of leaves <= 0
            alg_frame["rmse_diff"] = (np.sqrt(np.abs(alg_frame["train_mse"])) / np.sqrt(np.abs(dframes[0]["train_mse"]))) - 1
            alg_frame["rmse_diff_abs"] = alg_frame["train_mse"] - dframes[0]["train_mse"]
            alg_frame["rmse_makes_sense"] = np.logical_and(dframes[0]["train_mse"] > 0, alg_frame["leaves"] > 0)
            alg_frame["terminal_diff"] = (alg_frame["terminal_calls"] / dframes[0]["terminal_calls"])
            alg_frame["terminal_makes_sense"] = np.logical_and(dframes[0]["terminal_calls"] > 0, alg_frame["Algorithm"] != "OSRT", alg_frame["leaves"] > 0)
            # alg_frame.drop(alg_frame[alg_frame["leaves"] <= 0].index, inplace=True)
    return pd.concat(frames, ignore_index=True)

def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for f in fig_formats:
        plt.savefig(str(path) + f)
    plt.close()

def plot_mse(frame, path):
    valids = frame[frame["rmse_makes_sense"]]
    print("Plotting RMSE diff")
    plot = sns.lineplot(x="depth", y="rmse_diff", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Percentage difference in RMSE compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(path)

    print("Plotting RMSE abs diff")
    plot = sns.lineplot(x="depth", y="rmse_diff_abs", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Difference in MSE compared to STreeD (None)")
    save_plot(Path(str(path) + "_abs"))

def plot_runtime(frame, path):
    valids = frame[frame["leaves"] > 0]
    print("Plotting runtime")
    plot = sns.lineplot(x="depth", y="time", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Training time (s)")
    plot.set_yscale("log")
    save_plot(path)

def plot_terminal_calls(frame, path):
    valids = frame[frame["terminal_makes_sense"]]
    print("Plotting terminal calls")
    plot = sns.lineplot(x="depth", y="terminal_diff", hue="Algorithm", style="Algorithm", data=valids)
    plot.set(xlabel="Depth", ylabel="Percentage calls to depth-2 solver compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(path)

def plot_ecdf_runtime(frame, path, timeout):
    print("Plotting ECDF of runtime")
    if (frame["time"] < 0).any():
        print("Output contains error, not plotting anything")
        #return # Allow temporarily
    plot = sns.ecdfplot(x="time", hue="Algorithm", log_scale=True, data=frame)
    plot.set(xlabel="Time", ylabel="Percentage trees solved")
    plot.axes.set(xlim=(0.0001, timeout))
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(path)

if __name__ == "__main__":
    datasets_path = "./data/datasets.txt"
    if not os.path.exists(datasets_path):
        print("./data/datasets.txt not found. Run ./prepare_data.py to generate datasets\n")
        exit()
    
    datasets = []
    with open(datasets_path) as datasets_file:
        datasets.extend([f.strip()[:-4] for f in datasets_file.readlines()])

    sns.set_style({"font.family": "Arial"})
    sns.set_style(style="darkgrid")
    sns.color_palette("colorblind")

    combined_df = read_all(datasets, algs)

    # for ds in datasets:
    #     fig_path = Path(f'./figures/mse_diff/{ds}')
    #     plot_mse(combined_df[combined_df["Dataset"] == ds], fig_path)

    # for ds in datasets:
    #     fig_path = Path(f'./figures/time/{ds}')
    #     plot_runtime(combined_df[combined_df["Dataset"] == ds], fig_path)
    
    plot_mse(combined_df, Path("./figures/mse"))
    plot_terminal_calls(combined_df, Path("./figures/terminal"))
    plot_ecdf_runtime(combined_df, Path("./figures/runtime_ecdf"), 30)