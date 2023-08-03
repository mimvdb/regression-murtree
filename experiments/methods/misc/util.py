from pathlib import Path
import pandas as pd
import json

SCRIPT_DIR = Path(__file__).parent.resolve()
PREPARED_DIR = SCRIPT_DIR / ".." / ".." / ".." / "data" / "prepared"
PREFIX_DATA = PREPARED_DIR / "all"

# Script to load dataframes for different methods
# S = Split, F = Fit, X = Split/Fit, CONT = numerical, CAT = nominal, BIN = numerical binned binarized, BIN_CAT = nominal one-hot
# |F| = |CONT|, |F_b| = |BIN| + |BIN_CAT|
########## CONT ### CAT ### BIN ### BIN_CAT
# CART:    S        -       -       S
# IAI:     S        -       -       S
# DTIP:    S        -       -       S
# ORT:     S        -       -       S
# GUIDE:   S        S       -       -
# STREED:  -        -       S       S
# OSRT:    -        -       S       S
# GUIDE-L: X        S       -       -
# IAI-L:   X        -       -       S
# ORT-L:   X        -       -       X
# STREED-L:F        -       S       S


def load_data_info(name):
    with open(PREPARED_DIR / (name + ".json"), "r") as df_info_file:
        df_info = json.load(df_info_file)
    return df_info


def load_data_cont_bincat(name):
    df_info = load_data_info(name)
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), names=df_info["cols"])
    X = df.loc[:, df_info["continuous_cols"] + df_info["bincat_cols"]]
    y = df.iloc[:, 0]
    return X, y, df_info


def load_data_bin_bincat(name):
    df_info = load_data_info(name)
    df = pd.read_csv(PREFIX_DATA / (name + ".csv"), names=df_info["cols"])
    X = df.loc[:, df_info["binary_cols"]]
    y = df.iloc[:, 0]
    return X, y, df_info
