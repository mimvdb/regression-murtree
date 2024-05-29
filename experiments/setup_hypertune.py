#! /usr/bin/env python
# creates the experiment file for running the hypertuning experiments

from pathlib import Path
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).parent.resolve()
DEPTH = 5
DEPTH_L = DEPTH - 1
TIMEOUT = 60 * 15
TUNE = True

def generate_experiments():
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    experiments = []

    for info in infos:
        if info["filename"] in ["household"]: continue
        
        if "splits" not in info:
            print(f"Skipping dataset: {info['filename']}. It does not contain splits")
            continue
        for split in info["splits"]:
            lr = {
                "method": "lr",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
            }

            cart = {
                "method": "cart",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "tune": TUNE
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
                "tune": TUNE,
                "use_kmeans": 1,
                "use_task_bound": 1,
                "use_lower_bound": 1,
                "use_d2": 1,
            }
            osrt = {
                "method": "osrt",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0.00001,
                "tune": TUNE,
            }
            iai = {
                "method": "iai",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data": split["train"],
                "test_data": split["test"],
                "tune": TUNE,
                "complexity_penalty": 0.0
            }

            guide_sl = {
                "method": "guide_sl",
                "timeout": TIMEOUT,
                "depth": DEPTH_L,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            guide_l = {
                "method": "guide_l",
                "timeout": TIMEOUT,
                "depth": DEPTH_L,
                "train_data": split["train"],
                "test_data": split["test"],
            }
            streed_pwsl = {
                "method": "streed_pwsl",
                "timeout": TIMEOUT,
                "depth": DEPTH_L,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0,
                "ridge": 0,
                "tune": TUNE,
                "use_d2": 1,
            }
            streed_pwl = {
                "method": "streed_pwl",
                "timeout": TIMEOUT,
                "depth": DEPTH_L,
                "train_data": split["train"],
                "test_data": split["test"],
                "complexity_penalty": 0,
                "lasso": 0,
                "ridge": 0,
                "tune": TUNE,
            }
            iai_l = {
                "method": "iai_l",
                "timeout": TIMEOUT,
                "depth": DEPTH_L,
                "train_data": split["train"],
                "test_data": split["test"],
                "tune": TUNE,
                "complexity_penalty": 0.0
            }
            ort = {
                    "method": "ort",
                    "timeout": TIMEOUT,
                    "depth": DEPTH,
                    "train_data":  split["train"],
                    "test_data": split["test"],
                    "complexity_penalty": 0,
                    "linear": False,
                    "lasso_penalty": 0,
                    "metric": "MAE"
                }
            ort_l = {
                    "method": "ort",
                    "timeout": TIMEOUT,
                    "depth": DEPTH_L,
                    "train_data":  split["train"],
                    "test_data": split["test"],
                    "complexity_penalty": 0,
                    "linear": True,
                    "lasso_penalty": 0,
                    "metric": "MAE"
                }
            dtip = {
                "method": "dtip",
                "timeout": TIMEOUT,
                "depth": DEPTH,
                "train_data":  split["train"],
                "test_data": split["test"],
            }

            experiments.append(lr)
            experiments.append(cart)
            experiments.append(guide)
            experiments.append(guide_sl)
            experiments.append(guide_l)
            experiments.append(streed_pwc)
            experiments.append(streed_pwsl)
            experiments.append(streed_pwl)
            experiments.append(osrt)
            experiments.append(iai)
            experiments.append(iai_l)
            
            experiments.append(ort)
            experiments.append(ort_l)
            experiments.append(dtip)

    # Randomize experiment order so no methods gets an unfair advantage on average
    random.shuffle(experiments)
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup experiments")
    parser.add_argument("--file", default=str(SCRIPT_DIR / "experiments.json"))
    args = parser.parse_args()

    experiments = generate_experiments()

    print(f"Generated {len(experiments)} experiments.")

    with open(args.file, "w") as experiments_file:
        json.dump(experiments, experiments_file, indent=4)
