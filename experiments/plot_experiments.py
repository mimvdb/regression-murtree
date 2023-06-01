from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

if __name__ == "__main__":
    algs = ["streed_none", "streed_all", "streed_equivalent", "streed_similarity", "osrt"]
    alg_name = {
        "streed_none": "STreeD (None)",
        "streed_all": "STreeD (k-Means + Similarity)",
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

    frames = []
    for dataset in datasets:
        dframes = []
        for alg in algs:
            alg_frame = pd.read_csv(f"./results/{dataset}/{alg}.csv")
            alg_frame["Algorithm"] = alg_name[alg]
            frames.append(alg_frame)
            dframes.append(alg_frame)
            alg_frame["mse_diff"] = (alg_frame["train_mse"] / dframes[0]["train_mse"]) - 1
            if alg != "osrt": alg_frame["terminal_diff"] = (alg_frame["terminal_calls"] / dframes[0]["terminal_calls"])
            alg_frame.drop(alg_frame[alg_frame["leaves"] <= 0].index, inplace=True)
        combined_df = pd.concat(dframes, ignore_index=True)

        plot = sns.lineplot(x="depth", y="time", hue="Algorithm", style="Algorithm", data=combined_df)
        plot.set(xlabel="Depth", ylabel="Training time [s]")

        fig_path = Path(f'./figures/time/{dataset}.png')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path))
        plt.close()

        plot = sns.lineplot(x="depth", y="terminal_diff", hue="Algorithm", style="Algorithm", data=combined_df[combined_df["Algorithm"] != "OSRT"])
        plot.set(xlabel="Depth", ylabel="Percentage different in Terminal calls compared to STreeD (None)")

        fig_path = Path(f'./figures/terminal/{dataset}.png')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path))
        plt.close()


    combined_df = pd.concat(frames, ignore_index=True)
    plot = sns.lineplot(x="depth", y="mse_diff", hue="Algorithm", style="Algorithm", data=combined_df)
    plot.set(xlabel="Depth", ylabel="Percentage difference in MSE compared to STreeD (None)")

    fig_path = Path(f'./figures/mse_diff/{".".join(datasets)}.png')
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path))
    plt.close()