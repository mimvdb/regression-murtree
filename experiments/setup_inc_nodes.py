#! /usr/bin/env python
# creates the experiment file for running the hypertuning experiments

from pathlib import Path
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
MIN_DEPTH = 2
MAX_DEPTH = 8
TIMEOUT = 60 * 15

def generate_experiments():
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    experiments = []

    for info in infos:
        if info["filename"] not in ["airfoil"]: continue
        if "splits" not in info:
            print(f"Skipping dataset: {info['filename']}. It does not contain splits")
            continue
        for depth in range(MIN_DEPTH, MAX_DEPTH+1):
            for split in info["splits"]:
                lr = {
                    "method": "lr",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                }

                cart = {
                    "method": "cart",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["train"],
                    "tune": False
                }
                guide = {
                    "method": "guide",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                }
                streed_pwc = {
                    "method": "streed_pwc",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["train"],
                    "complexity_penalty": 0,
                    "tune": False,
                    "use_kmeans": 1,
                    "use_task_bound": 1,
                    "use_lower_bound": 1,
                    "use_d2": 1,
                }
                osrt = {
                    "method": "osrt",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                    "complexity_penalty": 0.0001,
                    "tune": True,
                }
                iai = {
                    "method": "iai",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["train"],
                    "tune": False
                }

                guide_l = {
                    "method": "guide_l",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                }
                streed_pwl = {
                    "method": "streed_pwl",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                    "complexity_penalty": 0,
                    "lasso": 0.999,
                    "tune": True,
                }
                iai_l = {
                    "method": "iai_l",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": split["train"],
                    "test_data": split["test"],
                }
                
                # experiments.append(lr)
                experiments.append(cart)
                #experiments.append(guide)
                # experiments.append(guide_l)
                experiments.append(streed_pwc)
                # experiments.append(streed_pwl)
                # experiments.append(osrt)
                experiments.append(iai)
                # experiments.append(iai_l)

    # Randomize experiment order so no methods gets an unfair advantage on average
    random.shuffle(experiments)
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup experiments")
    parser.add_argument("--file", default=str(SCRIPT_DIR / "experiments.json"))
    args = parser.parse_args()

    experiments = generate_experiments()

    with open(args.file, "w") as experiments_file:
        json.dump(experiments, experiments_file, indent=4)
