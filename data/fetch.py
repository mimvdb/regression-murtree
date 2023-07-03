#! /usr/bin/env python

from pathlib import Path
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile
import json
import pandas as pd
import urllib.request

SCRIPT_DIR = Path(__file__).parent.resolve()

def fetch_auto_mpg():
    url = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
    resp = urllib.request.urlopen(url)
    zipfile = ZipFile(BytesIO(resp.read()))
    names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
    df = pd.read_csv(zipfile.open('auto-mpg.data'), sep='\s+', na_values=['?'], header=None, names=names)
    return df

datasets = [
    {"name": "Auto MPG", "filename": "auto_mpg", "fetch": fetch_auto_mpg}
]

# Fetch and save all datasets, the first column is the label
def save_all(dir):
    infos = []

    for ds in datasets:
        info = {"name": ds["name"], "filename": ds["filename"]}
        frame: pd.DataFrame = ds["fetch"]()
        info["instances_raw"] = frame.shape[0]
        info["features_raw"] = frame.shape[1]
        frame.to_csv(dir / (ds["filename"] + ".csv"), header=True, index=False)
        infos.append(info)
    
    with open(dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    save_all(SCRIPT_DIR / "raw")