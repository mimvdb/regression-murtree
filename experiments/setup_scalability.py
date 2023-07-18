#! /usr/bin/env python
# creates the datasets and experiment file for running the scalability experiments

from pathlib import Path
import json
import argparse
import random

SCRIPT_DIR = Path(__file__).parent.resolve()


def generate_experiments():
    experiments = []

    for depth in range(1, 5):
        for _ in range(5):
            streed = {
                "method": "streed",
                "timeout": 60,
                "depth": depth,
                "train_data": str(
                    SCRIPT_DIR / "old" / "data" / "streed" / "airfoil.csv"
                ),
                "test_data": str(
                    SCRIPT_DIR / "old" / "data" / "streed" / "airfoil.csv"
                ),
                "complexity_penalty": 0.0001,
                "tune": False,
                "use_kmeans": 1,
                "use_task_bound": 1,
                "use_lower_bound": 1,
                "use_d2": 1,
            }
            osrt = {
                "method": "osrt",
                "timeout": 60,
                "depth": depth,
                "train_data": str(SCRIPT_DIR / "old" / "data" / "osrt" / "airfoil.csv"),
                "test_data": str(SCRIPT_DIR / "old" / "data" / "osrt" / "airfoil.csv"),
                "complexity_penalty": 0.0001,
                "tune": False,
            }
            experiments.append(streed)
            experiments.append(osrt)

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
