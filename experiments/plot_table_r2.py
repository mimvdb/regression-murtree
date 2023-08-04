#! /usr/bin/env python

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Out of sample table generator",
        description="Generate out of sample table from results",
    )
    parser.add_argument("--in-file", default=str(SCRIPT_DIR / "results.csv"))
    args = parser.parse_args()

    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    frame = pd.read_csv(args.in_file)

    methods = ["LR", "CART", "GUIDE", "IAI", "OSRT", "STreeD (PWC)", "GUIDE-L", "IAI-L", "STreeD (PWL)"]
    method_map = {
        "LR": "lr",
        "CART": "cart",
        "GUIDE": "guide",
        "GUIDE-L": "guide_l",
        "OSRT": "osrt",
        "STreeD (PWC)": "streed_pwc",
        "STreeD (PWL)": "streed_pwl",
        "IAI": "iai",
        "IAI-L": "iai_l"
    }

    for info in infos:
        print(f"{info['name']} &")
        for method in methods:
            sep = "\\\\" if method == methods[-1] else "&"
            split_results = []
            num_fail = 0
            num_timeout = 0
            unknown = 0
            for split in info["splits"]:
                filter = np.logical_and(frame["method"] == method_map[method], frame["train_data"] == split["train"])
                filtered = frame[filter]
                if len(filtered) != 1:
                    unknown = 1
                    break
                timeout = filtered["timeout"].values[0]
                time = filtered["time"].values[0]
                test_r2 = filtered["test_r2"].values[0]

                if time > timeout:
                    num_timeout += 1

                if test_r2 != -1:
                    split_results.append(test_r2)
                else:
                    num_fail += 1
            if unknown == 0:
                result = "-" if len(split_results) == 0 else f"{np.mean(split_results):.2f}"
                print(f"{result} {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
            else:
                print(f"? {sep} % {method}")