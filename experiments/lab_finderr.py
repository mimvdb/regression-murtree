#! /usr/bin/env python
# Runs a set of experiments in the specified file

from pathlib import Path
import csv
import json
import argparse
import os
from typing import List

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Experiment error aggregator",
        description="Aggregates the errors from multiple sync runs",
    )
    parser.add_argument("--in-dir", default=str(SCRIPT_DIR / "experiment"))
    parser.add_argument("--out-file", default=str(SCRIPT_DIR / "log.err"))
    args = parser.parse_args()

    errors = []
    for (dirpath, _, filenames) in os.walk(args.in_dir):
        print(f"Reading directory {dirpath}")
        for file in filenames:
            if not file.endswith(".err"): continue
            print(f"Reading errors: {file}")
            with open(Path(dirpath) / file, "r") as err_file:
                errors.append(err_file.read())


    with open(args.out_file, "w") as f:
        f.write("\n".join(errors))