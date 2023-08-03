#! /usr/bin/env python
# creates the experiment file for running the hypertuning experiments

from pathlib import Path
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
DEPTH = 1 # 5
TIMEOUT = 60

def generate_experiments():
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    experiments = []

    for info in infos:
        if info["filename"] == "household": continue
        for split in info["splits"]:
            cart = {
                "method": "cart",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            guide = {
                "method": "guide",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            streed_pwc = {
                "method": "streed_pwc",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0,
                "tune": True,
                "use_kmeans": 1,
                "use_task_bound": 1,
                "use_lower_bound": 1,
                "use_d2": 1,
            }
            streed_pwl = {
                "method": "streed_pwl",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0,
                "lasso": 0.999,
                "tune": True,
            }
            osrt = {
                "method": "osrt",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0.0001,
                "tune": True,
            }
            iai = {
                "method": "iai",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            iai_l = {
                "method": "iai_l",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            
            # experiments.append(cart)
            # experiments.append(guide)
            # experiments.append(streed_pwc)
            # experiments.append(streed_pwl)
            experiments.append(osrt)
            # experiments.append(iai)
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
