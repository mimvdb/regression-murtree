#! /usr/bin/env python
# Runs a set of experiments in the specified file

from methods.streed import run_streed
from pathlib import Path
import csv
import json
import os
import sys
import argparse
from typing import List

SCRIPT_DIR = Path(__file__).parent.resolve()
OSRT_PATH = SCRIPT_DIR / ".." / ".." / "optimal-sparse-regression-tree-public" / "build" / "gosdt" # SCRIPT_DIR / "gosdt"
STREED_PATH = SCRIPT_DIR / ".." / ".." / "streed2" / "build" / "STREED" # SCRIPT_DIR / "STREED"

def run_experiments(experiments: List):
    results = []

    for e in experiments:
        if e["method"] == "streed":
            result = run_streed(
                str(STREED_PATH),
                e["timeout"],
                e["depth"],
                e["train_data"],
                e["test_data"],
                e["complexity_penalty"],
                e["use_kmeans"],
                e["use_task_bound"],
                e["use_lower_bound"],
                e["use_d2"],
            )
            result["method"] = f'streed_kmeans{e["use_kmeans"]}_tasklb{e["use_task_bound"]}_lb{e["use_lower_bound"]}_terminal{e["use_d2"]}'
        
        result["timeout"] = e["timeout"]
        result["depth"] = e["depth"]
        result["train_data"] = e["train_data"]
        result["test_data"] = e["test_data"]
        result["complexity_penalty"] = e["complexity_penalty"]
        results.append(result)
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Synchronous experiment runner", description="Runs, parses, and saves output of multiple experiments sequentially"
    )
    parser.add_argument("--in-file", default=str(SCRIPT_DIR / "experiments.json"))
    parser.add_argument("--out-file", default=str(SCRIPT_DIR / "results.csv"))
    args = parser.parse_args()

    with open(args.in_file, "r") as experiments_file:
        experiments = json.load(experiments_file)

    results = run_experiments(experiments)
    attributes = ["method", "timeout", "depth", "train_data", "test_data", "complexity_penalty", "time", "train_mse", "test_mse", "leaves", "terminal_calls"]

    with open(args.out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(attributes)

        for run in results:
            row = [run[attribute] for attribute in attributes]
            writer.writerow(row)