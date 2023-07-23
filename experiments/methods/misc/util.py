from pathlib import Path
import pandas as pd
import json

SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARED_DIR = SCRIPT_DIR / ".." / ".." / ".." / "data" / "prepared"
PREFIX_DATA = PREPARED_DIR / "streed_pwl"


def load_data_continuous(name):
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), sep=" ", header=None)
    with open(PREPARED_DIR / (name + ".json"), "r") as df_info_file:
        df_info = json.load(df_info_file)
    X = df.iloc[:, 1:1+df_info["continuous_features"]]
    y = df.iloc[:, 0]
    return X, y, df_info


def load_data_binary(name):
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), sep=" ", header=None)
    with open(PREPARED_DIR / (name + ".json"), "r") as df_info_file:
        df_info = json.load(df_info_file)
    X = df.iloc[:, 1+df_info["continuous_features"]:]
    y = df.iloc[:, 0]
    return X, y, df_info


def load_data_info(name):
    with open(PREPARED_DIR / (name + ".json"), "r") as df_info_file:
        df_info = json.load(df_info_file)
    return df_info