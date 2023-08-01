from pathlib import Path
import pandas as pd
import json

SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARED_DIR = SCRIPT_DIR / ".." / ".." / ".." / "data" / "prepared"
PREFIX_DATA = PREPARED_DIR / "all"


def load_data_info(name):
    with open(PREPARED_DIR / (name + ".json"), "r") as df_info_file:
        df_info = json.load(df_info_file)
    return df_info


def load_data_continuous_categorical(name):
    df_info = load_data_info(name)
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), names=df_info["cols"])
    X = df.loc[:, df_info["continuous_cols"] + df_info["categorized_cols"]]
    y = df.iloc[:, 0]
    return X, y, df_info


def load_data_binary(name):
    df_info = load_data_info(name)
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), names=df_info["cols"])
    X = df.loc[:, df_info["binary_cols"]]
    y = df.iloc[:, 0]
    return X, y, df_info
