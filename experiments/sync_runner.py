#! /usr/bin/env python
# Runs a set of experiments in the specified file

from methods.guide import run_guide
from methods.cart import run_cart
from methods.streed import run_streed_pwc, run_streed_pwl
from methods.osrt import run_osrt
from methods.ort import run_ort
from methods.dtip import run_dtip
from methods.iai import run_iai, run_iai_l

# from methods.iai import run_iai
from pathlib import Path
import csv
import json
import argparse
from typing import List

SCRIPT_DIR = Path(__file__).parent.resolve()
OSRT_PATH = (
    SCRIPT_DIR
    / ".."
    / ".."
    / "optimal-sparse-regression-tree-public"
    / "build"
    / "osrt"
)  # SCRIPT_DIR / "osrt"
STREED_PATH = (
    SCRIPT_DIR / ".." / ".." / "streed2" / "build" / "STREED"
)  # SCRIPT_DIR / "STREED"
GUIDE_PATH = SCRIPT_DIR / "methods" / "misc" / "guide"


def run_experiments(experiments: List):
    results = []
    n_experiments = len(experiments)
    for e_i, e in enumerate(experiments):
        print(f"{e_i+1}/{n_experiments}: ", e)
        if e["method"] == "streed_pwc":
            result = run_streed_pwc(
                str(STREED_PATH),
                e["timeout"],
                e["depth"],
                e["train_data"],
                e["test_data"],
                e["complexity_penalty"],
                e["tune"],
                e["use_kmeans"],
                e["use_task_bound"],
                e["use_lower_bound"],
                e["use_d2"],
            )
            result["method"] = "streed_pwc"
            if not e["tune"]:
                result[
                    "method"
                ] += f'_kmeans{e["use_kmeans"]}_tasklb{e["use_task_bound"]}_lb{e["use_lower_bound"]}_terminal{e["use_d2"]}'
        elif e["method"] == "streed_pwl":
            result = run_streed_pwl(
                str(STREED_PATH),
                e["timeout"],
                e["depth"],
                e["train_data"],
                e["test_data"],
                e["complexity_penalty"],
                e["lasso"],
                e["tune"],
            )
            result["method"] = f"streed_pwl"
        elif e["method"] == "osrt":
            result = run_osrt(
                str(OSRT_PATH),
                e["timeout"],
                e["depth"],
                e["train_data"],
                e["test_data"],
                e["complexity_penalty"],
                e["tune"],
            )
            result["method"] = "osrt"
        elif e["method"] == "ort":
            result = run_ort(
                e["timeout"],
                e["depth"],
                e["train_data"],
                e["test_data"],
                e["complexity_penalty"],
                e["linear"],
                e["lasso_penalty"],
                e["metric"],
            )
            result["method"] = f'ort_l{e["linear"]}_metric{e["metric"]}'
        elif e["method"] == "dtip":
            result = run_dtip(e["timeout"], e["depth"], e["train_data"], e["test_data"])
            result["method"] = "dtip"
        elif e["method"] == "cart":
            result = run_cart(e["timeout"], e["depth"], e["train_data"], e["test_data"])
            result["method"] = "cart"
        elif e["method"] == "guide":
            result = run_guide(str(GUIDE_PATH), e["timeout"], e["depth"], e["train_data"], e["test_data"])
            result["method"] = "guide"
        elif e["method"] == "iai":
            result = run_iai(e["timeout"], e["depth"], e["train_data"], e["test_data"])
            result["method"] = "iai"
        elif e["method"] == "iai_l":
            result = run_iai_l(e["timeout"], e["depth"], e["train_data"], e["test_data"])
            result["method"] = "iai_l"

        result["timeout"] = e["timeout"]
        result["depth"] = e["depth"]
        result["train_data"] = e["train_data"]
        result["test_data"] = e["test_data"]

        # Only note down complexity penalty to disambiguate runs, might be different than actual e.g. for hypertuning runs
        result["complexity_penalty"] = (
            e["complexity_penalty"] if "complexity_penalty" in e else 0.0
        )

        results.append(result)
        print(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Synchronous experiment runner",
        description="Runs, parses, and saves output of multiple experiments sequentially",
    )
    parser.add_argument("--in-file", default=str(SCRIPT_DIR / "experiments.json"))
    parser.add_argument("--out-file", default=str(SCRIPT_DIR / "results.csv"))
    args = parser.parse_args()

    with open(args.in_file, "r") as experiments_file:
        experiments = json.load(experiments_file)

    results = run_experiments(experiments)
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

    results.sort(
        key=lambda v: (
            v["method"],
            v["train_data"],
            v["test_data"],
            v["depth"],
            v["complexity_penalty"],
        )
    )

    with open(args.out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(attributes)

        for run in results:
            row = [run[attribute] for attribute in attributes]
            writer.writerow(row)
