#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).parent.resolve()


def save_all_formats(dir, info, frame, filename):
    osrt_base = dir / "osrt"
    streed_pwc_base = dir / "streed_pwc"
    streed_pwl_base = dir / "streed_pwl"
    all_base = dir / "all"

    osrt_base.mkdir(parents=True, exist_ok=True)
    streed_pwc_base.mkdir(exist_ok=True)
    streed_pwl_base.mkdir(exist_ok=True)
    all_base.mkdir(exist_ok=True)

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

    with open(dir / (filename + ".json"), "w") as data_info:
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
        print(f"Preparing data: {info['name']}")
        frame = pd.read_csv(bin_dir / (info['filename'] + ".csv"))
        save_all_formats(prep_dir, info, frame, info["filename"])

        # Make 5 train/test splits
        disc = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile", subsample=20000, random_state=42)
        y_ = disc.fit_transform(frame.iloc[:, 0].values.reshape(-1, 1))

        splits = []
        for fold_index in range(5):
            print(f"Fold: {fold_index}")
            test_percentage = 0.2
            train, test = train_test_split(frame, test_size=test_percentage, random_state = 42 + fold_index, stratify=y_)
            train_file = f"{info['filename']}_train_{fold_index}"
            test_file  = f"{info['filename']}_test_{fold_index}"

            save_all_formats(prep_dir, info, train, train_file)
            save_all_formats(prep_dir, info, test, test_file)
            splits.append({"train": train_file, "test": test_file})

        info["splits"] = splits
            

    with open(prep_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)
        


if __name__ == "__main__":
    prepare_all(SCRIPT_DIR / "binarized", SCRIPT_DIR / "prepared")