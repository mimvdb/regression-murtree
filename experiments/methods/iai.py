import os

os.environ["IAI_DISABLE_COMPILED_MODULES"] = "True"
from interpretableai import iai
import pandas as pd
import numpy as np


def run_iai(timeout, depth, train_data, test_data):
    train_df = pd.read_csv(train_data, sep=" ", header=None)
    test_df = pd.read_csv(test_data, sep=" ", header=None)

    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(max_depth=depth),
        cp=[0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001],
    )
    grid.fit(X_train, y_train)
    reg = grid.get_learner()
    return {
        "time": -1,
        "train_mse": -1,
        "test_mse": -1,
        "leaves": -1,
        "terminal_calls": -1,
    }


def run_iai_l(timeout, depth, train_data, test_data):
    train_df = pd.read_csv(train_data, sep=" ", header=None)
    test_df = pd.read_csv(test_data, sep=" ", header=None)

    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            max_depth=0,
            minbucket=X_train.shape[1] * 10,
            regression_features={"All"},
        ),
        regression_lambda=[0.1, 0.01, 0.001, 0.0001],
    )
    grid.fit(X_train, y_train)
    starting_lambda = grid.get_best_params()["regression_lambda"]

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            max_depth=depth,
            minbucket=X_train.shape[1] * 10,
            regression_features={"All"},
            regression_lambda=starting_lambda,
        ),
        cp=[0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001],
    )
    grid.fit(X_train, y_train)
    best_cp = grid.get_best_params()["cp"]

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            max_depth=depth,
            minbucket=X_train.shape[1] * 10,
            cp=best_cp,
            regression_features={"All"},
        ),
        regression_lambda=[0.0001, 0.001, 0.01, 0.1],
    )
    grid.fit(X_train, y_train)

    reg = grid.get_learner()
    return {
        "time": -1,
        "train_mse": -1,
        "test_mse": -1,
        "leaves": -1,
        "terminal_calls": -1,
    }
