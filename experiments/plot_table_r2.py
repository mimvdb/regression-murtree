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

    methods = ["CART", "GUIDE", "STreeD"]
    method_map = {"CART": "cart", "GUIDE": "guide", "STreeD": "streed_pwc"}

    for info in infos:
        print(f"{info['name']} &")
        for method in methods:
            sep = "\\\\" if method == methods[-1] else "&"
            split_results = []
            for split in info["splits"]:
                filter = np.logical_and(frame["method"] == method_map[method], frame["train_data"] == split["train"])
                split_results.append(frame[filter]["test_r2"].values[0])
            result = np.mean(split_results)
            print(f"{result} {sep} % {method}")
