import os

os.environ["IAI_DISABLE_COMPILED_MODULES"] = "True"
from interpretableai import iai
from methods.misc.util import load_data_continuous_categorical


def run_iai(timeout, depth, train_data, test_data):
    X_train, y_train, train_info = load_data_continuous_categorical(train_data)
    X_test, y_test, test_info = load_data_continuous_categorical(test_data)

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(max_depth=depth),
        cp=[0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001],
    )
    grid.fit(X_train, y_train)
    reg = grid.get_learner()
    return {
        "time": -1,
        "train_r2": -1,
        "test_r2": -1,
        "leaves": -1,
        "terminal_calls": -1,
    }


def run_iai_l(timeout, depth, train_data, test_data):
    X_train, y_train, train_info = load_data_continuous_categorical(train_data)
    X_test, y_test, test_info = load_data_continuous_categorical(test_data)

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            max_depth=0,
            minbucket=X_train.shape[1] * 10,
            regression_features=set(train_info["continuous_cols"]),
        ),
        regression_lambda=[0.1, 0.01, 0.001, 0.0001],
    )
    grid.fit(X_train, y_train)
    starting_lambda = grid.get_best_params()["regression_lambda"]

    grid = iai.GridSearch(
        iai.OptimalTreeRegressor(
            max_depth=depth,
            minbucket=X_train.shape[1] * 10,
            regression_features=set(train_info["continuous_cols"]),
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
            regression_features=set(train_info["continuous_cols"]),
        ),
        regression_lambda=[0.0001, 0.001, 0.01, 0.1],
    )
    grid.fit(X_train, y_train)

    reg = grid.get_learner()
    return {
        "time": -1,
        "train_r2": -1,
        "test_r2": -1,
        "leaves": -1,
        "terminal_calls": -1,
    }
