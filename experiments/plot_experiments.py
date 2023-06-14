#! /usr/bin/env python

from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import mean_squared_error
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

def read_all(datasets, algs):
    frames = []
    for dataset in datasets:
        for alg in algs:
            print(f"Reading data for {dataset} - {alg}")
            alg_frame = pd.read_csv(f"./results/{dataset}/{alg}.csv")
            alg_frame["algorithm"] = alg
            alg_frame["dataset"] = dataset
            alg_frame["sequence"] = 0 # Old data has no repeats
            # alg_frame["Complexity cost"] = pd.cut(alg_frame["cost_complexity"], cost_complexity_bins)
            # cost_categories = alg_frame["Complexity cost"].cat.categories
            frames.append(alg_frame)
            
    return pd.concat(frames, ignore_index=True)

def preprocess(frame: pd.DataFrame, dataset_variances):
    # Split into a frame for each algorithm
    alg_frames = {}
    for alg in algs:
        alg_frames[alg] = frame[frame["algorithm"] == alg].copy()
        alg_frames[alg]["dataset_var"] = alg_frames[alg]["dataset"].map(dataset_variances)
        alg_frames[alg]["Training objective"] = alg_frames[alg]["train_mse"] + \
            alg_frames[alg]["cost_complexity"] * (alg_frames[alg]["leaves"] - 1) * alg_frames[alg]["dataset_var"]
        alg_frames[alg]["Algorithm"] = alg_name[alg]
        alg_frames[alg]["validity_test"] = True
    
    # Line up all matching rows and augment data
    id_set = ["dataset", "depth", "cost_complexity", "sequence"]
    for alg in algs:
        alg_frames[alg].sort_values(by=id_set)
        alg_frames[alg].reset_index(drop=True, inplace=True)
        for id_attr in id_set:
            # Works because streed_none is first in algs
            alg_frames[alg]["validity_test"] &= (alg_frames[alg][id_attr] == alg_frames["streed_none"][id_attr])
            if not alg_frames[alg]["validity_test"].all():
                raise Exception("Some value does not line up")

        # np.abs to get rid of warning for negatives. Invalids get filtered out by x_makes_sense
        alg_frames[alg]["rmse_diff"] = (np.sqrt(np.abs(alg_frames[alg]["train_mse"])) / np.sqrt(np.abs(alg_frames["streed_none"]["train_mse"]))) - 1
        alg_frames[alg]["rmse_diff_abs"] = alg_frames[alg]["train_mse"] - alg_frames["streed_none"]["train_mse"]
        alg_frames[alg]["training_diff"] = (alg_frames[alg]["Training objective"] / alg_frames["streed_none"]["Training objective"]) - 1
        alg_frames[alg]["rtraining_diff"] = (np.sqrt(np.abs(alg_frames[alg]["Training objective"])) / np.sqrt(np.abs(alg_frames["streed_none"]["Training objective"]))) - 1
        alg_frames[alg]["training_diff_abs"] = alg_frames[alg]["Training objective"] - alg_frames["streed_none"]["Training objective"]
        alg_frames[alg]["rmse_makes_sense"] = np.logical_and(alg_frames["streed_none"]["train_mse"] > 0, alg_frames[alg]["leaves"] > 0)
        alg_frames[alg]["terminal_diff"] = (alg_frames[alg]["terminal_calls"] / alg_frames["streed_none"]["terminal_calls"])
        alg_frames[alg]["terminal_makes_sense"] = np.logical_and(alg_frames["streed_none"]["terminal_calls"] > 0, alg_frames[alg]["Algorithm"] != "OSRT", alg_frames[alg]["leaves"] > 0)
    print("Preprocessing done")
    return pd.concat(alg_frames.values(), ignore_index=True)


def save_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for f in fig_formats:
        plt.savefig(str(path) + f)
    plt.close()

def plot_mse(frame: pd.DataFrame, path):
    valids = frame[frame["rmse_makes_sense"]]
    valids[np.logical_and(np.abs(frame["rtraining_diff"]) > 0.05, frame["depth"] <= 2)].to_csv("./bad_values.csv")
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

    print("Plotting training diff")
    plot = sns.lineplot(x="depth", y="training_diff", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Percentage difference in RMSE compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(Path(str(path) + "_train"))

    print("Plotting rtraining diff")
    plot = sns.lineplot(x="depth", y="rtraining_diff", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Percentage difference in RMSE compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(Path(str(path) + "_rtrain"))

    print("Plotting training abs diff")
    plot = sns.lineplot(x="depth", y="training_diff_abs", hue="Algorithm", style="Algorithm",
                        #size="Complexity cost", size_order=reversed(cost_categories),
                        markers=True, dashes=True, data=valids)
    plot.set(xlabel="Depth", ylabel="Difference in MSE compared to STreeD (None)")
    save_plot(Path(str(path) + "_train_abs"))

    # print("Plotting Root Training Objective Difference")
    # plot = sns.boxenplot(x="Algorithm", y="rtraining_diff", hue="Algorithm", data=valids)
    # plot.set(xlabel="Depth", ylabel="Percentage difference in RMSE compared to STreeD (None)")
    # plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    # save_plot(Path(str(path) + "_bar"))

    print("Plotting Root Training Objective Difference")
    plot = sns.scatterplot(x="Training objective", y="training_diff", hue="Algorithm", style="Algorithm",
                        markers=True, data=valids)
    plot.set(xlabel="Training objective", ylabel="Percentage difference in training objective compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    plot.set_xscale("log")
    save_plot(Path(str(path) + "_smalllargeerror"))

    print("Plotting RMSE Difference")
    plot = sns.scatterplot(x="train_mse", y="rmse_diff", hue="Algorithm", style="Algorithm",
                        markers=True, data=valids)
    plot.set(xlabel="MSE", ylabel="Percentage difference in RMSE compared to STreeD (None)")
    plot.axes.yaxis.set_major_formatter(PercentFormatter(1))
    plot.set_xscale("log")
    save_plot(Path(str(path) + "_smalllargeerror_mse"))

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

    dataset_variances = {}
    for ds in datasets:
        df = pd.read_csv(f"./data/osrt/{ds}.csv")
        y_train = df[df.columns[-1]].to_numpy()
        dataset_variances[ds] = mean_squared_error(y_train, np.full(len(y_train), np.mean(y_train)))

    sns.set_style({"font.family": "Arial"})
    sns.set_style(style="darkgrid")
    sns.color_palette("colorblind")

    combined_df = read_all(datasets, algs)
    combined_df = preprocess(combined_df, dataset_variances)

    # for ds in datasets:
    #     fig_path = Path(f'./figures/mse_diff/{ds}')
    #     plot_mse(combined_df[combined_df["dataset"] == ds], fig_path)

    # for ds in datasets:
    #     fig_path = Path(f'./figures/time/{ds}')
    #     plot_runtime(combined_df[combined_df["dataset"] == ds], fig_path)
    
    plot_mse(combined_df, Path("./figures/mse"))
    plot_terminal_calls(combined_df, Path("./figures/terminal"))
    plot_ecdf_runtime(combined_df, Path("./figures/runtime_ecdf"), 30)