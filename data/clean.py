#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()

def clean_all(raw_dir, clean_dir):
    with open(raw_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    for info in infos:
        frame = pd.read_csv(raw_dir / (info['filename'] + ".csv"))

        removed_cols = []
        categorized_cols = []
        for col in frame:
            if pd.api.types.is_numeric_dtype(frame[col].dtypes):
                continue
            codes, uniques = pd.factorize(frame[col])
            n_uniques = len(uniques)
            if n_uniques <= 1 or n_uniques >= 20:
                removed_cols.append(col)
            else:
                frame[col] = codes
                frame[col].replace(-1, np.nan)
                categorized_cols.append(col)
        frame.drop(columns=removed_cols, inplace=True)
        info["removed_cols"] = removed_cols
        info["categorized_cols"] = categorized_cols

        frame.dropna(inplace=True) # Drop NA rows after removing columns
        info["instances_cleaned"] = frame.shape[0]
        info["features_cleaned"] = frame.shape[1]

        frame.to_csv(clean_dir / (info["filename"] + ".csv"), header=True, index=False)
    
    with open(clean_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    clean_all(SCRIPT_DIR / "raw", SCRIPT_DIR / "cleaned")