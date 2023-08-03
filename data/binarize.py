#! /usr/bin/env python

from pathlib import Path
import json
import copy
import pandas as pd
import numpy as np
from sklearn.base import check_array
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree

SCRIPT_DIR = Path(__file__).parent.resolve()
BINS = 10

# Find duplicate columns without transposing https://stackoverflow.com/a/32961145/3395956
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for _, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            if cs[i] in dups: continue
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                if cs[j] in dups: continue
                ja = vs.iloc[:, j].values
                if np.allclose(ia, ja, equal_nan=True):
                    dups.append(cs[j])
                    break

    return dups

def all_same_columns(frame):
    sames = []
    for jj in range(frame.shape[1]):
            column = frame.iloc[:, jj]
            column_name = frame.columns[jj]
            if len(column.unique()) == 1:
                sames.append(column_name)
    return sames

class CategoricalToBinary:
    
    def __init__(self, org_column_name, values):
        self.org_column_name = org_column_name
        self.unique_values = values.unique()
        print(f"{org_column_name}, uniques: {self.unique_values}")
        if len(self.unique_values) == 2: # binary categorical variable
            self.values = [self.unique_values[0]]
        else:
            self.values = self.unique_values
        self.column_names = [f"{org_column_name}_cat_{val}" for val in self.values]

    def transform(self, X):
        for i, new_column_name in enumerate(self.column_names):
            X[new_column_name] = (X[self.org_column_name] == self.values[i]).astype(int)
        return X

    def update_info(self, info):
        if not "binary_cols" in info:
            info["binary_cols"] = []
        info["binary_cols"].extend(self.column_names)

class ContinuousToBinary:

    def __init__(self, org_column_name, values, y, n_bins):
        self.org_column_name = org_column_name
        if len(np.unique(values)) <=  n_bins + 1:
            uniques = np.unique(values)
            self.cutpoints = [(uniques[i] + uniques[i+1]) / 2 for i in range(len(uniques) - 1)]
        else:
            reg = DecisionTreeRegressor(max_leaf_nodes=n_bins + 1)
            reg.fit(values.values.reshape(-1, 1), y)
            self.cutpoints = sorted([t for t in reg.tree_.threshold if t != sklearn.tree._tree.TREE_UNDEFINED])
        counts = []
        for j, cut_point in enumerate(self.cutpoints):
            counts.append((values >= cut_point).astype(int).sum())
        print(f"{self.org_column_name} cuts: {self.cutpoints}")
        print(f"{self.org_column_name} Counts: {counts}")

    def transform(self, X):
        for j, cut_point in enumerate(self.cutpoints):
            col_name = f"{self.org_column_name}_bin_{j}"
            series = X[self.org_column_name]
            X[col_name] = (series >= cut_point).astype(int)

    def update_info(self, info):
        if not "binary_cols" in info:
            info["binary_cols"] = []
        for j, cut_point in enumerate(self.cutpoints):
            col_name = f"{self.org_column_name}_bin_{j}"
            info["binary_cols"].append(col_name)

        if not "cuts" in info:
            info["cuts"] = {}
        info["cuts"][self.org_column_name] = self.cutpoints
        
class Discretizer:

    def __init__(self, n_bins=5):
        self.redundant_columns = []
        self.redundant_binary_columns = []
        self.transformations = {}
        self.n_bins = n_bins   

    def fit(self, X, y, info):
        self.redundant_columns = duplicate_columns(X)
        self.redundant_columns.extend(all_same_columns(X))
        if len(self.redundant_columns) > 0:
            print("Drop redunant columns: ", ", ".join(self.redundant_columns))

        check_array(X, dtype="numeric")

        n_features = X.shape[1]
        for jj in range(n_features):
            column = X.iloc[:, jj]
            column_name = X.columns[jj]
            if column_name in self.redundant_columns: continue
            
            if column_name in info["categorized_cols"]:
                self.transformations[column_name] = CategoricalToBinary(column_name, column)
            else:
                self.transformations[column_name] = ContinuousToBinary(column_name, column, y, self.n_bins)

        X2 = X.copy()
        info2 = copy.deepcopy(info)
        X2 = self.transform(X2, info2)
        self.redundant_binary_columns = duplicate_columns(X2.loc[:, info2["binary_cols"]])
        if len(self.redundant_binary_columns) > 0:
            print("Drop redunant binary columns: ", ", ".join(self.redundant_binary_columns))

        return self
                

    def transform(self, X, info):
        n_features = X.shape[1] 
        for jj in range(n_features):
            column_name = X.columns[jj]
            if column_name in self.redundant_columns: continue
            transformation = self.transformations[column_name]
            transformation.transform(X)
            transformation.update_info(info)
        
        X.drop(self.redundant_columns, axis="columns", inplace=True)
        info["removed_cols"].extend(self.redundant_columns)
        X.drop(self.redundant_binary_columns, axis="columns", inplace=True)
        info["removed_cols"].extend(self.redundant_binary_columns)

        return X

def binarize(bin_dir, info, train_frame, train_name, test_frame=None, test_name=None):
    X = train_frame.iloc[:, 1:]
    y = train_frame.iloc[:, 0]
    train_info = copy.deepcopy(info)
    org_columns = list(X.columns)

    discretizer = Discretizer(BINS).fit(X, y, train_info)
    X = discretizer.transform(X, train_info)
    frame_out = pd.concat((y, X), axis=1)
    train_info["binary_cols"] = [s for s in train_info["binary_cols"] if s not in discretizer.redundant_binary_columns]
    train_info["continuous_cols"] = [s for s in org_columns if s not in train_info["categorized_cols"]]
    train_info["instances_binarized"] = frame_out.shape[0]
    train_info["features_binarized"] = frame_out.shape[1]
    
    frame_out.to_csv(bin_dir / (train_name + ".csv"), header=True, index=False)
    with open(bin_dir / (train_name + ".json"), "w") as data_info:
        json.dump(train_info, data_info, indent=4)

    if test_frame is None or test_name is None: return
    
    X = test_frame.iloc[:, 1:]
    y = test_frame.iloc[:, 0]
    test_info = copy.deepcopy(info)
    X = discretizer.transform(X, test_info)
    frame_out = pd.concat((y, X), axis=1)
    test_info["binary_cols"] = [s for s in test_info["binary_cols"] if s not in discretizer.redundant_binary_columns]
    test_info["continuous_cols"] = [s for s in org_columns if s not in test_info["categorized_cols"]]
    test_info["instances_binarized"] = frame_out.shape[0]
    test_info["features_binarized"] = frame_out.shape[1]

    frame_out.to_csv(bin_dir / (test_name + ".csv"), header=True, index=False)
    with open(bin_dir / (test_name + ".json"), "w") as data_info:
        json.dump(test_info, data_info, indent=4)

def binarize_all(split_dir, bin_dir):
    if not bin_dir.exists():
        bin_dir.mkdir(exist_ok=True)

    with open(split_dir / "info.json", "r") as info_json:
        infos = json.load(info_json)

    for info in infos:
        #if info["filename"] != 'optical': continue
        if info["filename"] == 'household': continue

        print(f"****** Binarizing {info['filename']} ******")
        frame = pd.read_csv(split_dir / (info["filename"] + ".csv"))

        binarize(bin_dir, copy.deepcopy(info), frame, info["filename"])

        for sp_ix, split in enumerate(info["splits"]):
            print(f"****** Binarizing {info['filename']} - Fold {sp_ix+1}  ******")
            train = split["train"]
            test = split["test"]
            
            train_frame = pd.read_csv(split_dir / (train + ".csv"))
            test_frame = pd.read_csv(split_dir / (test + ".csv"))

            binarize(bin_dir, copy.deepcopy(info), train_frame, train, test_frame, test)


        

    with open(bin_dir / "info.json", "w") as info_json:
        json.dump(infos, info_json, indent=4)


if __name__ == "__main__":
    binarize_all(SCRIPT_DIR / "split", SCRIPT_DIR / "binarized")
