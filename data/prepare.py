#! /usr/bin/env python

from pathlib import Path
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()

def prepare_all(bin_dir: Path, prep_dir: Path):
    with open(bin_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    osrt_base = prep_dir / "osrt"
    osrt_base.mkdir(parents=True, exist_ok=True)

    streed_pwc_base = prep_dir / "streed_pwc"
    streed_pwc_base.mkdir(exist_ok=True)

    streed_pwl_base = prep_dir / "streed_pwl"
    streed_pwl_base.mkdir(exist_ok=True)

    for info in infos:
        frame = pd.read_csv(bin_dir / (info['filename'] + ".csv"))

        X_cont = frame.loc[:,info["continuous_cols"]]
        X_bin = frame.loc[:,info["binary_cols"]]
        y = frame.iloc[:,0]

        osrt_frame = pd.concat([X_bin, y], axis="columns")
        streed_pwc_frame = pd.concat([y, X_bin], axis="columns")
        streed_pwl_frame = pd.concat([y, X_cont, X_bin], axis="columns")

        osrt_frame.to_csv(osrt_base / (info["filename"] + ".csv"), header=True, index=False)
        streed_pwc_frame.to_csv(streed_pwc_base / (info["filename"] + ".csv"), sep=" ", header=False, index=False)
        streed_pwl_frame.to_csv(streed_pwl_base / (info["filename"] + ".csv"), sep=" ", header=False, index=False)

        with open(prep_dir / (info["filename"] + ".json"), "w") as data_info:
            json.dump({}, data_info) # TODO write metadata from file, variance for R2, number of cont features for pwl etc.


if __name__ == "__main__":
    prepare_all(SCRIPT_DIR / "binarized", SCRIPT_DIR / "prepared")