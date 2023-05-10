import json
from pathlib import Path
from matplotlib import pyplot as plt
# Parts of this code were taken from https://github.com/ruizhang1996/regression-tree-benchmark
def get_alg_plotting_style(alg):
    if alg == 'streed':
        shape = 's'
        color = 'hotpink'
        alpha = 0.75
    elif alg == 'osrt':
        shape = '^'
        color = 'royalblue'
        alpha = 0.75
    else:
        shape = 'x'
        color = 'black'
        alpha = 0.75

    return shape, color, alpha

if __name__ == "__main__":
    with open("./results/experiment.json", "r") as experiment_output:
        results = json.load(experiment_output)
        for dataset_result in results:
            dataset = dataset_result["dataset"]
            dataset_results = dataset_result["results"]

            fig = plt.figure(figsize=(8, 5.5), dpi=80)
            plt.rc('font', size=18)
            x_max = 0
            xs = []
            ys = {
                "osrt": [],
                "streed": []
            }
            for depth_result in dataset_results:
                depth = depth_result["depth"]
                experiment = depth_result["experiment"]
                xs.append(depth)
                for alg in ["osrt", "streed"]:
                    ys[alg].append(experiment[alg]["time"])
            
            for alg in ["osrt", "streed"]:
                shape, color, alpha = get_alg_plotting_style(alg)
                plt.errorbar(xs, ys[alg], label=alg,
                        marker=shape, markersize=10, c=color, alpha=alpha, linewidth=1, linestyle='none')

            # if x_max > 15:
            #     x_major_locator = MultipleLocator(5)
            # else:
            #     x_major_locator = MultipleLocator(int(x_max / 4))

            # plt.gca().xaxis.set_major_locator(x_major_locator)
            # plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

            plt.xlabel('Max depth')
            plt.ylabel('Training Time (s)')
            plt.title('Training Time vs Tree depth\n ' + dataset)
            # if dataset == 'yacht' and depth == 9:
            #     plt.legend(loc='upper right')
            # if depth == 2:
            #     plt.legend(loc='upper left')
            plt.legend(loc='best')
            plt.tight_layout()
            fig_path = Path(f'./figures/time/{dataset}.png')
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(fig_path))
            plt.close()