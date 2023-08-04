#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


SCRIPT_DIR = Path(__file__).parent.resolve()

def save(dir, info, frame, filename):
    frame.to_csv(dir / (filename + ".csv"), header=True, index=False)

def split_all(clean_dir: Path, split_dir: Path):
    if not split_dir.exists():
        split_dir.mkdir(exist_ok=True)

    with open(clean_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)
    
    for info in infos:
        # if info["filename"] == 'household': continue
        print(f"****** Splitting {info['filename']} ******")
        frame = pd.read_csv(clean_dir / (info["filename"] + ".csv"))
        save(split_dir, info, frame, info["filename"])

        
        disc = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile", subsample=20000, random_state=42)
        ydisc = disc.fit_transform(frame.iloc[:, 0].values.reshape(-1, 1))

        splits = []
        for fold_index in range(5):
            print(f"Fold: {fold_index}")
            test_percentage = 0.2
            train, test = train_test_split(frame, test_size=test_percentage, random_state = 42 + fold_index, stratify=ydisc)
            train_file = f"{info['filename']}_train_{fold_index}"
            test_file  = f"{info['filename']}_test_{fold_index}"
            save(split_dir, info, train, train_file)
            save(split_dir, info, test,  test_file)

            splits.append({"train": train_file, "test": test_file})

        info["splits"] = splits
    
    with open(split_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    split_all(SCRIPT_DIR / "cleaned", SCRIPT_DIR / "split")