#! /usr/bin/env python

from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    with open(SCRIPT_DIR / ".." / "data" / "prepared" / "info.json", "r") as info_json:
        infos = json.load(info_json)

    rows = []

    for info in infos:

        with open(SCRIPT_DIR / ".." / "data" / "prepared" / f"{info['filename']}.json", "r") as info_json:
            file_info = json.load(info_json)

        rows.append(f"{info['name']} & {file_info['instances']} & {len(file_info['continuous_cols'])} & {len(file_info['binary_cols'])}")

    print("\\\\\n".join(sorted(rows)))