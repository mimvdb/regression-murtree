#! /usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import json

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    frame = pd.read_csv(SCRIPT_DIR / "results.csv")

    methods = ["CART", "GUIDE", "STreeD (PWC)", "STreeD (PWL)", "IAI", "IAI-L"]
    method_map = {"CART": "cart", "GUIDE": "guide", "STreeD (PWC)": "streed_pwc", "STreeD (PWL)": "streed_pwl", "IAI": "iai", "IAI-L": "iai_l"}

    for info in infos:
        print(f"{info['name']} &")
        for method in methods:
            sep = "\\\\" if method == methods[-1] else "&"
            split_results = []
            num_fail = 0
            num_timeout = 0
            for split in info["splits"]:
                filter = np.logical_and(frame["method"] == method_map[method], frame["train_data"] == split["train"])
                timeout = frame[filter]["timeout"].values[0]
                time = frame[filter]["time"].values[0]
                test_r2 = frame[filter]["test_r2"].values[0]

                if time > timeout:
                    num_timeout += 1

                if test_r2 != -1:
                    split_results.append(test_r2)
                else:
                    num_fail += 1
            result = "-" if len(split_results) == 0 else np.mean(split_results)
            print(f"{result} {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
