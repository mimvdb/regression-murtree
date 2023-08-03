#! /usr/bin/env python

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json
import random
import string
import pandas as pd
import urllib.request

SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / "cache"


def get_bytes(url):
    index_path = CACHE_DIR / "index.json"

    if not index_path.exists():
        index_path.parent.mkdir(exist_ok=True)
        with open(index_path, "w") as index_file:
            json.dump({}, index_file)

    with open(index_path, "r") as index_file:
        index = json.load(index_file)

    if url in index:
        with open(CACHE_DIR / index[url], "rb") as cached_file:
            return BytesIO(cached_file.read())

    resp = urllib.request.urlopen(url)
    result = resp.read()

    file_name = "".join(random.choices(string.ascii_lowercase, k=12)) + ".zip"
    file_path = CACHE_DIR / file_name
    if file_path.exists():
        raise RuntimeError(f"Cache error, {file_name} already exists")

    with open(file_path, "wb") as cached_file:
        cached_file.write(result)
        index[url] = file_name
    with open(index_path, "w") as index_file:
        json.dump(index, index_file, indent=4)

    return BytesIO(result)


def typecast_float(value):
    try:
        return float(value.replace(",", "."))
    except:
        return value


def fetch_auto_mpg():
    url = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
    zipfile = ZipFile(get_bytes(url))
    names = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model year",
        "origin",
        "car name",
    ]
    df = pd.read_csv(
        zipfile.open("auto-mpg.data"),
        sep="\s+",
        na_values=["?"],
        header=None,
        names=names,
    )
    return df


def fetch_airfoil():
    url = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
    zipfile = ZipFile(get_bytes(url))
    names = [
        "frequency",
        "angle_of_attack",
        "chord_length",
        "fs_velocity",
        "ssd_thickness",
        "sound_pressure",
    ]
    df = pd.read_csv(
        zipfile.open("airfoil_self_noise.dat"),
        sep="\s+",
        header=None,
        names=names,
    )
    sorted_names = names[-1:] + names[:-1]  # Bring label to front
    df = df.reindex(columns=sorted_names)
    return df


def fetch_auction():
    url = "https://archive.ics.uci.edu/static/public/713/auction+verification.zip"
    zipfile = ZipFile(get_bytes(url))
    df = pd.read_csv(
        zipfile.open("data.csv"),
        sep=",",
        header=0,
    )
    names = df.columns.values.tolist()
    df.drop(columns=["verification.result"], inplace=True)  # Classification target
    sorted_names = names[-1:] + names[:-1]  # Bring label to front
    df = df.reindex(columns=sorted_names)
    return df


def fetch_optical():
    url = "https://archive.ics.uci.edu/static/public/449/optical+interconnection+network.zip"
    zipfile = ZipFile(get_bytes(url))
    df = pd.read_csv(
        zipfile.open("optical_interconnection_network.csv"),
        sep=";",
        decimal=",",
        header=0,
    )
    label_col = df.pop("Channel Utilization")
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_real_estate():
    url = "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip"
    zipfile = ZipFile(get_bytes(url))
    try:
        df = pd.read_excel(
            zipfile.open("Real estate valuation data set.xlsx"),
            decimal=",",
            header=0,
        )
    except:
        df = pd.read_excel(
            zipfile.open("Real estate valuation data set.xlsx"),
            converters={
                col: typecast_float
                for col in [
                    "X3 distance to the nearest MRT station",
                    "X5 latitude",
                    "X6 longitude",
                    "Y house price of unit area",
                ]
            },
            header=0,
        )
    df.drop(columns=["No"], inplace=True)  # Row numbering
    label_col = df.pop("Y house price of unit area")
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_seoul_bike():
    url = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
    zipfile = ZipFile(get_bytes(url))
    df = pd.read_csv(
        zipfile.open("SeoulBikeData.csv"),
        encoding="iso-8859-3",  # celcius sign
        sep=",",
        header=0,
    )
    label_col = df.pop("Rented Bike Count")
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_servo():
    url = "https://archive.ics.uci.edu/static/public/87/servo.zip"
    zipfile = ZipFile(get_bytes(url))
    names = ["motor", "screw", "pgain", "vgain", "class"]
    df = pd.read_csv(zipfile.open("servo.data"), sep=",", header=None, names=names)
    sorted_names = names[-1:] + names[:-1]  # Bring label to front
    df = df.reindex(columns=sorted_names)
    return df


def fetch_synch():
    url = (
        "https://archive.ics.uci.edu/static/public/607/synchronous+machine+data+set.zip"
    )
    zipfile = ZipFile(get_bytes(url))
    df = pd.read_csv(
        zipfile.open("synchronous machine.csv"),
        sep=";",
        decimal=",",
        header=0,
    )
    label_col = df.pop("If")
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_yacht():
    url = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
    zipfile = ZipFile(get_bytes(url))
    names = [
        "center_of_buoyancy",
        "prismatic_coef",
        "length_displacement_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
        "residuary_resistance",
    ]
    df = pd.read_csv(
        zipfile.open("yacht_hydrodynamics.data"), sep="\s+", header=None, names=names
    )
    sorted_names = names[-1:] + names[:-1]  # Bring label to front
    df = df.reindex(columns=sorted_names)
    return df


def fetch_energy_heat():
    url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    zipfile = ZipFile(get_bytes(url))
    try:
        df = pd.read_excel(
            zipfile.open("ENB2012_data.xlsx"),
            decimal=",",
            header=0,
        )
    except:
        df = pd.read_excel(
            zipfile.open("ENB2012_data.xlsx"),
            converters={
                col: typecast_float
                for col in ["X1", "X2", "X3", "X4", "X5", "X7", "Y1", "Y2"]
            },
            header=0,
        )
    df.drop(columns=["Y2"], inplace=True)  # Drop cooling load label
    label_col = df.pop("Y1")  # Heating load
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_energy_cool():
    url = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
    zipfile = ZipFile(get_bytes(url))
    try:
        df = pd.read_excel(
            zipfile.open("ENB2012_data.xlsx"),
            decimal=",",
            header=0,
        )
    except:
        df = pd.read_excel(
            zipfile.open("ENB2012_data.xlsx"),
            converters={
                col: typecast_float
                for col in ["X1", "X2", "X3", "X4", "X5", "X7", "Y1", "Y2"]
            },
            header=0,
        )

    df.drop(columns=["Y1"], inplace=True)  # Drop heating load label
    label_col = df.pop("Y2")  # Cooling load
    df.insert(0, label_col.name, label_col)  # Bring label to front
    return df


def fetch_household():
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    zipfile = ZipFile(get_bytes(url))
    df = pd.read_csv(
        zipfile.open("household_power_consumption.txt"),
        sep=";",
        na_values=["?"],
        header=0,
        dtype={
            "Date": "string",
            "Time": "string",
            "Global_active_power": float,
            "Global_reactive_power": float,
            "Voltage": float,
            "Global_intensity": float,
            "Sub_metering_1": float,
            "Sub_metering_2": float,
            "Sub_metering_3": float,
        },
    )

    date_parts = df["Date"].str.split("/")
    df["Month"] = date_parts.apply(lambda x: int(x[1]))
    df["Year"] = date_parts.apply(lambda x: int(x[2]))

    time_parts = df["Time"].str.split(":")
    df["Hour"] = time_parts.apply(lambda x: int(x[0]))

    # Drop sub_meterings, as these may be a proxy for the label
    df.drop(
        columns=["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"], inplace=True
    )

    # Bring label to front
    label_col = df.pop("Global_active_power")
    df.insert(0, label_col.name, label_col)
    return df


datasets = [
    {"name": "Auto MPG", "filename": "auto_mpg", "fetch": fetch_auto_mpg},
    {"name": "Airfoil", "filename": "airfoil", "fetch": fetch_airfoil},
    {"name": "Auction", "filename": "auction", "fetch": fetch_auction},
    {
        "name": "Optical Network",
        "filename": "optical",
        "fetch": fetch_optical,
    },
    {
        "name": "Real Estate Valuation",
        "filename": "real-estate",
        "fetch": fetch_real_estate,
    },
    {
        "name": "Seoul Bike Sharing",
        "filename": "seoul-bike",
        "fetch": fetch_seoul_bike,
    },
    {"name": "Servo", "filename": "servo", "fetch": fetch_servo},
    {"name": "Sync Machine", "filename": "sync", "fetch": fetch_synch},
    {"name": "Yacht", "filename": "yacht", "fetch": fetch_yacht},
    {
        "name": "Energy (Heating)",
        "filename": "enb-heat",
        "fetch": fetch_energy_heat,
    },
    {
        "name": "Energy (Cooling)",
        "filename": "enb-cool",
        "fetch": fetch_energy_cool,
    },
    {
        "name": "Household Power",
        "filename": "household",
        "fetch": fetch_household,
    },
]


# Fetch and save all datasets, the first column is the label
def save_all(dir):
    infos = []

    if not dir.exists():
        dir.mkdir(exist_ok=True)

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
