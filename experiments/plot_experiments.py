from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import os

if __name__ == "__main__":
    algs = ["streed_none", "streed_all", "streed_equivalent", "streed_similarity", "osrt"]
    alg_name = {
        "streed_none": "STreeD (None)",
        "streed_all": "STreeD (k-Means Equiv. Points + Similarity)",
        "streed_equivalent": "STreeD (Equivalent Points + Similarity)",
        "streed_similarity": "STreeD (Similarity)",
        "osrt": "OSRT"
    }
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

    cost_complexity_bins = [0, 0.003, 0.01, 0.5]

    frames = []
    for dataset in datasets:
        dframes = []
        for alg in algs:
            alg_frame = pd.read_csv(f"./results/{dataset}/{alg}.csv")
            alg_frame["Algorithm"] = alg_name[alg]
            alg_frame["Complexity cost"] = pd.cut(alg_frame["cost_complexity"], cost_complexity_bins)
            cost_categories = alg_frame["Complexity cost"].cat.categories
            frames.append(alg_frame)
            dframes.append(alg_frame)
            # np.abs to get rid of warning for negatives. They will get filtered out anyway by the drop of leaves <= 0
            alg_frame["rmse_diff"] = (np.sqrt(np.abs(alg_frame["train_mse"])) / np.sqrt(dframes[0]["train_mse"])) - 1
            if alg != "osrt": alg_frame["terminal_diff"] = (alg_frame["terminal_calls"] / dframes[0]["terminal_calls"])
            alg_frame.drop(alg_frame[alg_frame["leaves"] <= 0].index, inplace=True)
        combined_df = pd.concat(dframes, ignore_index=True)

        plot = sns.lineplot(x="depth", y="time", hue="Algorithm", style="Algorithm",
                            size="Complexity cost", size_order=reversed(cost_categories), markers=True, dashes=True, data=combined_df)
        plot.set(xlabel="Depth", ylabel="Training time (s)")
        plot.set_yscale("log")

        fig_path = Path(f'./figures/time/{dataset}.svg')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path))
        fig_path = Path(f'./figures/time/{dataset}.png')
        plt.savefig(str(fig_path))
        plt.close()

        plot = sns.lineplot(x="depth", y="terminal_diff", hue="Algorithm", style="Algorithm", data=combined_df[combined_df["Algorithm"] != "OSRT"])
        plot.set(xlabel="Depth", ylabel="Percentage of Terminal calls compared to STreeD (None)")
        plot.axes.yaxis.set_major_formatter(PercentFormatter(1))

        fig_path = Path(f'./figures/terminal/{dataset}.svg')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path))
        fig_path = Path(f'./figures/terminal/{dataset}.png')
        plt.savefig(str(fig_path))
        plt.close()


        #combined_df = pd.concat(frames, ignore_index=True)
        plot = sns.lineplot(x="depth", y="rmse_diff", hue="Algorithm", style="Algorithm",
                            size="Complexity cost", size_order=reversed(cost_categories), markers=True, dashes=True, data=combined_df)
        plot.set(xlabel="Depth", ylabel="Percentage difference in RMSE compared to STreeD (None)")
        plot.axes.yaxis.set_major_formatter(PercentFormatter(1))

        fig_path = Path(f'./figures/mse_diff/{".".join([dataset])}.svg')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path))
        fig_path = Path(f'./figures/mse_diff/{".".join([dataset])}.png')
        plt.savefig(str(fig_path))
        plt.close()