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
    parser.add_argument("--in-file", default=str(SCRIPT_DIR / "../results/results-train.csv"))
    args = parser.parse_args()

    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    frame = pd.read_csv(args.in_file)

    df = frame.copy()
    df["dataset"] = df["train_data"].str.rsplit("_", n=2, expand=True)[0]
    df.loc[((df["method"] == "iai") | (df["method"] == "iai_l")) & (df["time"] > df["timeout"] * 2), "test_r2"] = -1 # *2 because leniency for methods run locally
    means = df.groupby(["dataset", "method"])["train_r2"].mean()
    means = means.unstack('method')
    
    constant_means = means[[m for m in ["cart", "guide", "iai", "streed_pwc_kmeans1_tasklb1_lb1_terminal1", "osrt", "dtip", "ort_lFalse_metricMAE"] if m in means.columns]]
    multiple_linear_means = means[[m for m in ["guide_l", "iai_l", "ort_lTrue_metricMAE", "streed_pwl"] if m in means.columns]]
    simple_linear_means = means[[m for m in ["guide_sl", "streed_pwsl_terminal1"] if m in means.columns]]

    best_constant_scores = constant_means.max(axis=1)
    best_simple_linear_scores = simple_linear_means.max(axis=1)
    best_multiple_linear_scores = multiple_linear_means.max(axis=1)
    best_scores_map = {
        "constant": best_constant_scores,
        "simple": best_simple_linear_scores,
        "multiple": best_multiple_linear_scores
    }

    constant_methods = ["CART", "GUIDE", "IAI", "DTIP", "ORT", "OSRT", "SRT-C"]
    linear_methods = ["GUIDE-SL", "SRT-SL", "GUIDE-L", "IAI-L", "ORT-L", "SRT-L"]
    method_map = {
        "LR": "lr",
        "CART": "cart",
        "GUIDE": "guide",
        "GUIDE-SL": "guide_sl",
        "GUIDE-L": "guide_l",
        "DTIP": "dtip",
        "ORT": "ort_lFalse_metricMAE",
        "OSRT": "osrt",
        "SRT-C": "streed_pwc_kmeans1_tasklb1_lb1_terminal1",
        "SRT-SL": "streed_pwsl_terminal1",
        "ORT-L": "ort_lTrue_metricMAE",
        "SRT-L": "streed_pwl",
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
    method_category = {
        "LR": "multiple",
        "CART": "constant",
        "GUIDE": "constant",
        "GUIDE-SL": "simple",
        "GUIDE-L": "multiple",
        "DTIP": "constant",
        "ORT": "constant",
        "OSRT": "constant",
        "SRT-C": "constant",
        "SRT-SL": "simple",
        "ORT-L": "multiple",
        "SRT-L": "multiple",
        "IAI": "constant",
        "IAI-L": "multiple"
    }
    
    has_time_outs = ["IAI", "DTIP", "ORT", "OSRT", "IAI-L", "ORT-L", "SRT-L"]
    has_negatives = ["DTIP", "ORT", "ORT-L"]

    for (methods, print_info) in [(constant_methods, False), (linear_methods, False)]:
        
        wins_per_method = {method: 0 for method in methods}

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
                best_scores = best_scores_map[method_category[method]]
                back_spacer = "~~~" if method in has_time_outs else ""
                front_spacer = "~" if method in has_negatives else ""
                for split in info["splits"]:
                    filter = np.logical_and(frame["method"] == method_map[method], frame["train_data"] == split["train"])
                    filtered = frame[filter]
                    if len(filtered) != 1:
                        unknown = 1
                        continue
                    
                    timeout = filtered["timeout"].values[0]
                    time = filtered["time"].values[0]
                    train_r2 = filtered["train_r2"].values[0]

                    # Leniency for methods run locally
                    if method in ["IAI", "IAI-L"]: timeout *= 2

                    if time > timeout:
                        num_timeout += 1

                    if train_r2 != -1:
                        split_results.append(train_r2)
                    else:
                        num_fail += 1
                
                if num_timeout > 0 or num_fail > 0:
                    if num_timeout > 0 and len(split_results) > 0:
                        result = f"{np.mean(split_results): 4.2f}".replace(" ", "~")
                        if round(np.mean(split_results), 2) >= round(best_scores[info["filename"]], 2):
                            result = "\\textbf{" + result + "}"
                            wins_per_method[method] += 1
                        print(f"{result} * {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
                    elif num_timeout > 0:
                        print(f"~-{back_spacer} {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
                    else:
                        print(f"~OoM{back_spacer} {sep} % {method} (# Exceeded timeout: {num_timeout}, # Failed: {num_fail})")
                elif unknown == 1:
                    print(f"~?{back_spacer} {sep} % {method}")
                else:
                    result = f"{np.mean(split_results): 4.2f}".replace(" ", "~")
                    if round(np.mean(split_results), 2) >= round(best_scores[info["filename"]], 2):
                        result = "\\textbf{" + result + "}"
                        wins_per_method[method] += 1
                    print(f"{result}{back_spacer} {sep} % {method}")
        print("\midrule")

        print("Best &")
        for method in methods:
            sep = "\\\\" if method == methods[-1] else "&"
            print(f"{wins_per_method[method]} {sep} % {method}")

        print("\n\n\n")