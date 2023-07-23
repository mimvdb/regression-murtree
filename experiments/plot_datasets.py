#! /usr/bin/env python

from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    rows = []

    for info in infos:
        rows.append(f"{info['name']} & {info['instances_binarized']} & {len(info['continuous_cols'])} & {len(info['binary_cols'])}")

    print("\\\\\n".join(sorted(rows)))