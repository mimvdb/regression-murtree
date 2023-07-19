#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).parent.resolve()
BINS = 5


def binarize_all(clean_dir, bin_dir):
    if not bin_dir.exists(): bin_dir.mkdir(exist_ok=True)

    with open(clean_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    for info in infos:
        print(f"Binarizing {info['filename']}")
        frame = pd.read_csv(clean_dir / (info["filename"] + ".csv"))

        X = frame.iloc[:, 1:]
        y = frame.iloc[:, 0]

        discretizer = KBinsDiscretizer(BINS, strategy="quantile", subsample=None)
        discretizer.fit(X, y)

        binarized = []
        for i in range(1, frame.shape[1]):
            series = frame.iloc[:, i]
            series_name = frame.columns[i]
            cut_points = discretizer.bin_edges_[i - 1][
                1:-1
            ]  # Exclude label and left/right edges

            for j in range(len(cut_points)):
                frame[series_name + f"_bin_{j}"] = (series >= cut_points[j]) * 1
                print(
                    f"Uniques {series_name}_bin_{j}: {len(frame[series_name + f'_bin_{j}'].unique())}, Sum: {frame[series_name + f'_bin_{j}'].sum()}"
                )
            binarized.append(series_name)

        info["bins"] = list(map(lambda a: a.tolist(), discretizer.bin_edges_))

        frame = frame.T.drop_duplicates().T # Drop duplicate columns

        info["binarized"] = binarized
        info["instances_binarized"] = frame.shape[0]
        info["features_binarized"] = frame.shape[1]
        frame.to_csv(bin_dir / (info["filename"] + ".csv"), header=True, index=False)

    with open(bin_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    binarize_all(SCRIPT_DIR / "cleaned", SCRIPT_DIR / "binarized")
