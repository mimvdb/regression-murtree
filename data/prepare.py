#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).parent.resolve()


def save_all_formats(bin_dir, prep_dir, frame, filename):
    osrt_base = prep_dir / "osrt"
    streed_pwc_base = prep_dir / "streed_pwc"
    streed_pwl_base = prep_dir / "streed_pwl"
    all_base = prep_dir / "all"

    osrt_base.mkdir(parents=True, exist_ok=True)
    streed_pwc_base.mkdir(exist_ok=True)
    streed_pwl_base.mkdir(exist_ok=True)
    all_base.mkdir(exist_ok=True)

    with open(bin_dir / f"{filename}.json", "r") as info_json:
        info = json.load(info_json)

    X_cat = frame.loc[:,info["categorized_cols"]]
    X_cont = frame.loc[:,info["continuous_cols"]]
    X_bin = frame.loc[:,info["binary_cols"]]
    y = frame.iloc[:,0]

    osrt_frame = pd.concat([X_bin, y], axis="columns")
    streed_pwc_frame = pd.concat([y, X_bin], axis="columns")
    streed_pwl_frame = pd.concat([y, X_cont, X_bin], axis="columns")
    all_frame = pd.concat([y, X_cat, X_cont, X_bin], axis="columns")

    osrt_frame.to_csv(osrt_base / (filename + ".csv"), header=True, index=False)
    streed_pwc_frame.to_csv(streed_pwc_base / (filename + ".csv"), sep=" ", header=False, index=False)
    streed_pwl_frame.to_csv(streed_pwl_base / (filename + ".csv"), sep=" ", header=False, index=False)
    all_frame.to_csv(all_base / (filename + ".csv"), header=False, index=False)

    # Add GUIDE descriptor file
    with open(all_base / (filename + ".guide.in"), "w") as guide_file:
        guide_file.write(f"{filename}.csv\n")
        # Cleaned data cannot contain NA, but needs to be given
        guide_file.write(f"NA\n")
        # Rows start at line 1, there is no header (if this is set to 2, the R script does not assign the correct names to columns)
        guide_file.write(f"1\n")

        i = 1
        guide_file.write(f"{i} label d\n") # first is target label
        i += 1
        for col in streed_pwl_frame.iloc[:, 1:]:
            if col in info["categorized_cols"]:
                guide_file.write(f"{i} cat{i} c\n") # categorical, can be used in splitting
            elif col in info["continuous_cols"]:
                guide_file.write(f"{i} cont{i} n\n") # numerical, can be used in splitting or fitting
            elif col in info["binary_cols"]:
                guide_file.write(f"{i} bin{i} x\n") # excluded, don't use binarized variables
            else:
                guide_file.write(f"{i} unknown{i} ERROR\n") # should never happen
            i += 1

    with open(prep_dir / (filename + ".json"), "w") as data_info:
        json.dump({
            "binary_features": len(info["binary_cols"]),
            "continuous_features": len(info["continuous_cols"]),
            "mean_squared_error": mean_squared_error(y, np.full(len(y), np.mean(y))),
            "instances": len(y),
            "cols": all_frame.columns.tolist(),
            "binary_cols": info["binary_cols"],
            "continuous_cols": info["continuous_cols"],
            "categorized_cols": info["categorized_cols"],
            "bincat_cols": info["bincat_cols"]
        }, data_info, indent=4)


def prepare_all(bin_dir: Path, prep_dir: Path):
    with open(bin_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    for info in infos:
        if info["filename"] == 'household': continue
        print(f"Preparing data: {info['name']}")
        frame = pd.read_csv(bin_dir / (info['filename'] + ".csv"))
        save_all_formats(bin_dir, prep_dir, frame, info["filename"])

        # Make 5 train/test splits
        for sp_ix, split in enumerate(info["splits"]):
            print(f"Preparing data: {info['name']} - Fold {sp_ix+1}")
            train_file = split["train"]
            test_file = split["test"]
            train = pd.read_csv(bin_dir / (train_file + ".csv"))
            test = pd.read_csv(bin_dir / (test_file + ".csv"))
            save_all_formats(bin_dir, prep_dir, train, train_file)
            save_all_formats(bin_dir, prep_dir, test, test_file)            

    with open(prep_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)
        


if __name__ == "__main__":
    prepare_all(SCRIPT_DIR / "binarized", SCRIPT_DIR / "prepared")