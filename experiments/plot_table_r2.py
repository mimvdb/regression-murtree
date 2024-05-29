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
    parser.add_argument("--in-file", default=str(SCRIPT_DIR / "../results/results-tune.csv"))
    args = parser.parse_args()

    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    frame = pd.read_csv(args.in_file)

    df = frame.copy()
    df["dataset"] = df["train_data"].str.rsplit("_", n=2, expand=True)[0]
    df.loc[((df["method"] == "iai") | (df["method"] == "iai_l")) & (df["time"] > df["timeout"] * 2), "test_r2"] = -1 # *2 because leniency for methods run locally
    means = df.groupby(["dataset", "method"])["test_r2"].mean()
    means = means.unstack('method')
    
    constant_means = means[[m for m in ["cart", "guide", "iai", "streed_pwc", "osrt"] if m in means.columns]]
    linear_means = means[[m for m in ["lr", "guide_sl", "guide_l", "iai_l", "streed_pwsl", "streed_pwl_lasso", "streed_pwl_elasticnet"] if m in means.columns]]

    best_constant_scores = constant_means.max(axis=1)
    best_linear_scores = linear_means.max(axis=1)

    constant_methods = ["CART", "GUIDE", "IAI", "OSRT", "SRT-C"]
    linear_methods = ["LR", "GUIDE-SL", "SRT-SL", "GUIDE-L", "IAI-L",  "SRT-L (Lasso)", "SRT-L (Elastic Net)"]
    method_map = {
        "LR": "lr",
        "CART": "cart",
        "GUIDE": "guide",
        "GUIDE-SL": "guide_sl",
        "GUIDE-L": "guide_l",
        "OSRT": "osrt",
        "SRT-C": "streed_pwc",
        "SRT-SL": "streed_pwsl",
        "SRT-L (Lasso)": "streed_pwl_lasso",
        "SRT-L (Elastic Net)": "streed_pwl_elasticnet",
        "IAI": "iai",
        "IAI-L": "iai_l"
    }
    data_set_map = {
        "Airfoil Self-Noise": "Airfoil",
        "Auction Verification": "Auction",
        "Auto MPG": "Auto MPG",
        "Energy Efficiency (cooling load)": "Energy (C)",
        "Energy Efficiency (heating load)": "Energy (H)",
        "Household Power Consumption": "Household",
        "Optical Interconnection Network": "Optical Net.",
        "Real Estate Valuation": "Real Estate",
        "Seoul Bike Sharing Demand": "Seoul Bike",
        "Servo": "Servo",
        "Synchronous Machine": "Synch.",
        "Yacht Hydrodynamics": "Yacht"
    }

    for (methods, best_scores, print_info) in [(constant_methods, best_constant_scores, True), (linear_methods, best_linear_scores, False)]:
        for info in sorted(infos, key=lambda x: x["name"]):
            #if info['name'] == 'Household Power Consumption': continue
            print(f"{data_set_map[info['name']]} &")
            if print_info:
                with open(SCRIPT_DIR / ".." / "data" / "prepared" / f"{info['filename']}.json", "r") as more_info_json:
                    more_info = json.load(more_info_json)
                    print(f"{more_info['instances']} & % no. instances")
                    print(f"{more_info['continuous_features']} & % no. continuous features")
                    print(f"{more_info['binary_features']} & % no. binary features")
            
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
                        continue
                    
                    timeout = filtered["timeout"].values[0]
                    time = filtered["time"].values[0]
                    test_r2 = filtered["test_r2"].values[0]

                    # Leniency for methods run locally
                    if method in ["IAI", "IAI-L"]: timeout *= 2

                    if time > timeout:
                        num_timeout += 1

                    if test_r2 != -1:
                        split_results.append(test_r2)
                    else:
                        num_fail += 1
                
                if num_timeout > 0 or num_fail > 0:
                    print(f"- {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
                elif unknown == 1:
                    print(f"? {sep} % {method}")
                else:
                    result = f"{np.mean(split_results):.2f}"
                    if round(np.mean(split_results), 2) >= round(best_scores[info["filename"]], 2):
                        result = "\\textbf{" + result + "}"
                    print(f"{result} {sep} % {method}")
        print("\n\n\n")