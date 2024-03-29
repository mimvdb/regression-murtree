import os
import time
from sklearn.metrics import r2_score
from methods.misc.util import load_data_cont_bincat, load_data_bin_bincat



def run_iai(timeout, depth, train_data, test_data, tune, cp):
    # Import iai in function to allow importing this module without having it installed
    os.environ["IAI_DISABLE_COMPILED_MODULES"] = "True"
    from interpretableai import iai

    X_train, y_train, train_info = load_data_cont_bincat(train_data)
    X_test, y_test, test_info = load_data_cont_bincat(test_data)

    # IAI has trouble with some column names. E.g. dots are replaced by underscores which causes mismatches. Replace with indices
    orig_columns = X_train.columns.tolist()
    new_columns  = list(map(str,range(len(orig_columns))))
    X_train.columns = new_columns
    X_test.columns = new_columns

    start_time = time.time()  # Start timer after reading data

    if tune:
        grid = iai.GridSearch(
            iai.OptimalTreeRegressor(max_depth=depth),
            cp=[0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001],
        )
        grid.fit(X_train, y_train)
        reg = grid.get_learner()
    else:
        reg = iai.OptimalTreeRegressor(max_depth=depth, cp=cp)
        reg.fit(X_train, y_train)
        
    duration = time.time() - start_time
    train_r2 = r2_score(y_train, reg.predict(X_train))
    test_r2 = r2_score(y_test, reg.predict(X_test))
    leaves = int((reg.get_num_nodes() + 1) / 2)

    return {
        "time": duration,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "leaves": leaves,
        "terminal_calls": -1,
    }


def run_iai_l(timeout, depth, train_data, test_data, tune, cp):
    # Import iai in function to allow importing this module without having it installed
    os.environ["IAI_DISABLE_COMPILED_MODULES"] = "True"
    from interpretableai import iai

    X_train, y_train, train_info = load_data_cont_bincat(train_data)
    X_test, y_test, test_info = load_data_cont_bincat(test_data)

    # IAI has trouble with some column names. E.g. dots are replaced by underscores which causes mismatches. Replace with indices
    orig_columns = X_train.columns.tolist()
    new_columns  = list(map(str,range(len(orig_columns))))
    X_train.columns = new_columns
    X_test.columns = new_columns
    regression_cols = list(map(lambda x: str(orig_columns.index(x)), train_info["continuous_cols"]))

    start_time = time.time()  # Start timer after reading data

    if tune:
        grid = iai.GridSearch(
            iai.OptimalTreeRegressor(
                max_depth=0,
                minbucket=len(regression_cols) * 10,
                regression_features=regression_cols,
            ),
            regression_lambda=[0.1, 0.01, 0.001, 0.0001],
        )
        grid.fit(X_train, y_train)
        starting_lambda = grid.get_best_params()["regression_lambda"]

        grid = iai.GridSearch(
            iai.OptimalTreeRegressor(
                max_depth=depth,
                minbucket=len(regression_cols) * 10,
                regression_features=regression_cols,
                regression_lambda=starting_lambda,
            ),
            cp=[0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001],
        )
        grid.fit(X_train, y_train)
        best_cp = grid.get_best_params()["cp"]

        grid = iai.GridSearch(
            iai.OptimalTreeRegressor(
                max_depth=depth,
                minbucket=len(regression_cols) * 10,
                cp=best_cp,
                regression_features=regression_cols,
            ),
            regression_lambda=[0.0001, 0.001, 0.01, 0.1],
        )
        grid.fit(X_train, y_train)

        reg = grid.get_learner()
    else:
        reg = iai.OptimalTreeRegressor(max_depth=depth,
                                        cp=cp,
                                        minbucket=len(regression_cols) * 10,
                                        regression_features=regression_cols,
                                        regression_lambda=0)
        reg.fit(X_train, y_train)
    
    duration = time.time() - start_time
    train_r2 = r2_score(y_train, reg.predict(X_train))
    test_r2 = r2_score(y_test, reg.predict(X_test))
    leaves = (reg.get_num_nodes() + 1) / 2

    return {
        "time": duration,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "leaves": leaves,
        "terminal_calls": -1,
    }
