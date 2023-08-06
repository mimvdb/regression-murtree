#! /usr/bin/env python
# creates the experiment file for running the scalability experiments

from pathlib import Path
import csv
import json
import argparse
import random
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
TIMEOUT = 1000
CPS = [0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
REPEATS = 5
SKIP_DATASETS = ["household"]


def generate_experiments(depth, prev_timeouts):
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    experiments = []

    for info in infos:
        data = info["filename"]
        if data in SKIP_DATASETS: continue
        for _ in range(REPEATS):
            for cp in CPS:
                streed = {
                    "method": "streed_pwc",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": data,
                    "test_data": data,
                    "complexity_penalty": cp,
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
                    "train_data": data,
                    "test_data": data,
                    "complexity_penalty": cp,
                    "tune": False,
                }
                ort = {
                    "method": "ort",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": data,
                    "test_data": data,
                    "complexity_penalty": cp,
                    "linear": False,
                    "lasso_penalty": 0,
                    "metric": "MAE"
                }
                ort_l = {
                    "method": "ort",
                    "timeout": TIMEOUT,
                    "depth": depth,
                    "train_data": data,
                    "test_data": data,
                    "complexity_penalty": cp,
                    "linear": True,
                    "lasso_penalty": 0,
                    "metric": "MAE"
                }
                if ("streed_pwc_kmeans1_tasklb1_lb1_terminal1", data, cp) not in prev_timeouts: experiments.append(streed)
                if ("osrt", data, cp) not in prev_timeouts: experiments.append(osrt)
                if ("ort_lFalse_metricMAE", data, cp) not in prev_timeouts: experiments.append(ort)
                if ("ort_lTrue_metricMAE", data, cp) not in prev_timeouts: experiments.append(ort_l)

            dtip = {
                "method": "dtip",
                "timeout": TIMEOUT,
                "depth": depth,
                "train_data": data,
                "test_data": data
            }
            if ("dtip", data, 0.0) not in prev_timeouts: experiments.append(dtip)

    # Randomize experiment order so no methods gets an unfair advantage on average
    random.shuffle(experiments)
    return experiments

def write_timed_out(depth, prev_timeouts):
    # If lower depth timed out, higher depth will also timeout. Save compute by skipping.
    results = []
    for timed_out in prev_timeouts:
        (method, data, cp) = timed_out
        results.extend([{
            "method": method,
            "timeout": TIMEOUT,
            "depth": depth,
            "train_data": data,
            "test_data": data,
            "complexity_penalty": cp,
            "time": -1,
            "train_r2": -1,
            "test_r2": -1,
            "leaves": -1,
            "terminal_calls": -1,
        } for _ in range(REPEATS)])

    attributes = [
        "method",
        "timeout",
        "depth",
        "train_data",
        "test_data",
        "complexity_penalty",
        "time",
        "train_r2",
        "test_r2",
        "leaves",
        "terminal_calls",
    ]

    results.sort(key=lambda v: (v["method"], v["train_data"], v["test_data"], v["depth"], v["complexity_penalty"]))

    with open("predetermined_timeouts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(attributes)

        for run in results:
            row = [run[attribute] for attribute in attributes]
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup experiments")
    parser.add_argument("--prev", default=None)
    parser.add_argument("--out-file", default=str(SCRIPT_DIR / "experiments.json"))
    args = parser.parse_args()

    if args.prev is None:
        depth = 1
        prev_timeouts = set()
    else:
        old_results = pd.read_csv(args.prev)
        depth = int(old_results["depth"].max() + 1)
        has_timeout = old_results[old_results["time"] > old_results["timeout"]]
        prev_timeouts = {(x["method"], x["train_data"], round(x["complexity_penalty"], 4)) for x in has_timeout.to_records()}

    write_timed_out(depth, prev_timeouts)
    print(f"Wrote results for {len(prev_timeouts)} previous timeouts")

    experiments = generate_experiments(depth, prev_timeouts)
    print(f"Writing {len(experiments)} experiments to {args.out_file}")

    with open(args.out_file, "w") as experiments_file:
        json.dump(experiments, experiments_file, indent=4)