#! /usr/bin/env python
# creates the experiment file for running the scalability experiments

from pathlib import Path
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
TIMEOUT = 1000
REPEATS = 5
DEPTH = 5


def generate_experiments():
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info_scalability.json", "r") as info_json:
        infos = json.load(info_json)

    experiments = []

    for info in infos:
        data = info["filename"]
        for _ in range(REPEATS):
            streed = {
                "method": "streed_pwc",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": data,
                "test_data": data,
                "complexity_penalty": 0.0001,
                "tune": False,
                "use_kmeans": 1,
                "use_task_bound": 1,
                "use_lower_bound": 1,
                "use_d2": 1,
            }
            osrt = {
                "method": "osrt",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": data,
                "test_data": data,
                "complexity_penalty": 0.0001,
                "tune": False,
            }
            ort = {
                "method": "ort",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": data,
                "test_data": data,
                "complexity_penalty": 0.0001,
                "linear": False,
                "lasso_penalty": 0,
                "metric": "MAE"
            }
            ort_l = {
                "method": "ort",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": data,
                "test_data": data,
                "complexity_penalty": 0.0001,
                "linear": True,
                "lasso_penalty": 0,
                "metric": "MAE"
            }
            dtip = {
                "method": "dtip",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": data,
                "test_data": data
            }
            experiments.append(streed)
            experiments.append(osrt)
            experiments.append(ort)
            experiments.append(ort_l)
            experiments.append(dtip)

    # Randomize experiment order so no methods gets an unfair advantage on average
    random.shuffle(experiments)
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup experiments")
    parser.add_argument("--out-file", default=str(SCRIPT_DIR / "experiments.json"))
    args = parser.parse_args()

    experiments = generate_experiments()
    print(f"Writing {len(experiments)} experiments to {args.out_file}")

    with open(args.out_file, "w") as experiments_file:
        json.dump(experiments, experiments_file, indent=4)