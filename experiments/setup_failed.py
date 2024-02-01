#! /usr/bin/env python
# Filter out all experiments that have a result in results.csv

from pathlib import Path
import json
import argparse
import random
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()

def generate_experiments():
    with open(SCRIPT_DIR / "experiments.json") as experiments_json:
        experiments = json.load(experiments_json)
    results = pd.read_csv(SCRIPT_DIR / "results.csv").to_records()

    def get_key_exp(v):
        method = v["method"]
        if method == "streed_pwc":
            method += f'_kmeans{v["use_kmeans"]}_tasklb{v["use_task_bound"]}_lb{v["use_lower_bound"]}_terminal{v["use_d2"]}'
        elif method == "streed_pwsl":
            method += f'_terminal{v["use_d2"]}'
        elif method == "ort":
            method += f'_l{v["linear"]}_metric{v["metric"]}'
        return (method, v["train_data"], v["test_data"], v["depth"], v.get("complexity_penalty", 0.0))
    
    def get_key_res(v):
        return (v["method"], v["train_data"], v["test_data"], v["depth"], v["complexity_penalty"])

    experiments.sort(key=get_key_exp)

    i = 0
    j = 0
    end_i = len(experiments)
    end_j = len(results)

    new_experiments = []

    while i < end_i and j < end_j:
        key_exp = get_key_exp(experiments[i])
        key_res = get_key_res(results[j])
        # if key_exp != key_res: print(f"{key_exp} vs {key_res}")
        if key_exp == key_res:
            i += 1
            j += 1
        elif key_exp < key_res:
            new_experiments.append(experiments[i])
            i += 1
        else:
            print(f"WARN: Result that is not in experiments {key_res}")
            j += 1
    while i < end_i:
        new_experiments.append(experiments[i])
        i += 1
    
    print(f"Filtered out {len(experiments) - len(new_experiments)} experiments from {len(results)} results.")
    print(f"Resulting in {len(new_experiments)} new experiments from {len(experiments)}")

    # Randomize experiment order so no methods gets an unfair advantage on average
    random.shuffle(new_experiments)
    return new_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Setup experiments")
    parser.add_argument("--file", default=str(SCRIPT_DIR / "new_experiments.json"))
    args = parser.parse_args()

    experiments = generate_experiments()

    with open(args.file, "w") as experiments_file:
        json.dump(experiments, experiments_file, indent=4)
