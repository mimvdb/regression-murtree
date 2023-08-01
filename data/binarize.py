#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import OneHotEncoder

SCRIPT_DIR = Path(__file__).parent.resolve()
BINS = 10


# custom discretizer, since sklearn's doesn't work well for binary features.
class QuantileDiscretizer:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X):
        X = check_array(X, dtype="numeric")
        n_features = X.shape[1]
        n_bins = np.full(n_features, self.n_bins, dtype=int)

        bin_edges = np.zeros(n_features, dtype=object)
        for jj in range(n_features):
            column = X[:, jj]
            col_min, col_max = column.min(), column.max()

            if col_min == col_max:
                print("Feature %d is constant and will be replaced with 0." % jj)
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            quantiles = np.linspace(0, 100, n_bins[jj] + 1)
            bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            # This is the main change over KBinsDiscretizer. Add to_end=np.inf so that the last edge is never removed.
            # The last edge will be replaced with inf when transforming, so should be seen as different from the second to last edge.
            # Ensuring the last edge isn't removed allows binary features to correctly get two bins.
            # Remove bins whose width are too small (i.e., <= 1e-8)
            mask = np.ediff1d(bin_edges[jj][:-1], to_begin=np.inf, to_end=np.inf) > 1e-8
            bin_edges[jj] = bin_edges[jj][mask]
            if len(bin_edges[jj]) - 1 != n_bins[jj]:
                print(
                    "Bins whose width are too small (i.e., <= "
                    "1e-8) in feature %d are removed. Consider "
                    "decreasing the number of bins." % jj
                )
                n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

        return self


# Find duplicate columns without transposing https://stackoverflow.com/a/32961145/3395956
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for _, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if np.allclose(ia, ja, equal_nan=True):
                    dups.append(cs[i])
                    break

    return dups


def binarize_all(clean_dir, bin_dir):
    if not bin_dir.exists():
        bin_dir.mkdir(exist_ok=True)

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

        discretizer = QuantileDiscretizer(BINS)
        discretizer.fit(X)

        binary = []
        cuts = {}
        for i in range(1, frame.shape[1]):
            series = frame.iloc[:, i]
            series_name = frame.columns[i]

            # One-hot encode categories, discretize continuous.
            if series_name in info["categorized_cols"]:
                unique = series.unique()
                for val in unique:
                    col_name = series_name + f"_cat_{val}"
                    frame[col_name] = (series == val) * 1
                    binary.append(col_name)
                print(f"{series_name} uniques: {unique}")
            else:
                cut_points = discretizer.bin_edges_[i - 1][
                    1:-1
                ]  # Exclude label and left/right edges
                cuts[series_name] = cut_points.tolist()

                uniques = []
                counts = []
                for j in range(len(cut_points)):
                    col_name = series_name + f"_bin_{j}"
                    frame[col_name] = (series >= cut_points[j]) * 1
                    binary.append(col_name)
                    uniques.append(len(frame[col_name].unique()))
                    counts.append(frame[col_name].sum())
                print(f"{series_name} cuts: {cuts[series_name]}")
                print(f"Uniques: {uniques}")
                print(f"Counts: {counts}")

        info["cuts"] = cuts

        # Drop duplicate binarized columns
        dup = duplicate_columns(frame.loc[:, binary])
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
