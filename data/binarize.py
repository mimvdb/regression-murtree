#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).parent.resolve()
BINS = 10


# Find duplicate columns without transposing https://stackoverflow.com/a/32961145/3395956
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for _, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.allclose(ia, ja, equal_nan=True):
                    dups.append(cs[i])
                    break

    return dups


def binarize_all(clean_dir, bin_dir):
    if not bin_dir.exists(): bin_dir.mkdir(exist_ok=True)

    with open(clean_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    for info in infos:
        print(f"Binarizing {info['filename']}")
        frame = pd.read_csv(clean_dir / (info["filename"] + ".csv"))

        X = frame.iloc[:, 1:]

        # Drop duplicate continuous columns
        dup = duplicate_columns(X)
        info["removed_cols"].extend(dup)
        frame.drop(dup, axis="columns", inplace=True)

        X = frame.iloc[:, 1:]
        y = frame.iloc[:, 0]

        discretizer = KBinsDiscretizer(BINS, strategy="quantile", subsample=None)
        discretizer.fit(X, y)

        binary = []
        for i in range(1, frame.shape[1]):
            series = frame.iloc[:, i]
            series_name = frame.columns[i]
            cut_points = discretizer.bin_edges_[i - 1][
                1:-1
            ]  # Exclude label and left/right edges

            for j in range(len(cut_points)):
                col_name = series_name + f"_bin_{j}"
                frame[col_name] = (series >= cut_points[j]) * 1
                binary.append(col_name)
                print(
                    f"Uniques {col_name}: {len(frame[col_name].unique())}, Sum: {frame[col_name].sum()}"
                )

        info["bins"] = list(map(lambda a: a.tolist(), discretizer.bin_edges_))

        # Drop duplicate binarized columns
        dup = duplicate_columns(frame.loc[:,binary])
        info["removed_cols"].extend(dup)
        frame.drop(dup, axis="columns", inplace=True)

        info["binary_cols"] = [s for s in binary if s not in dup]
        info["continuous_cols"] = X.columns.tolist()
        info["instances_binarized"] = frame.shape[0]
        info["features_binarized"] = frame.shape[1]
        frame.to_csv(bin_dir / (info["filename"] + ".csv"), header=True, index=False)

    with open(bin_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    binarize_all(SCRIPT_DIR / "cleaned", SCRIPT_DIR / "binarized")
