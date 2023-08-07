from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from methods.misc.util import load_data_cont_bincat

SCRIPT_DIR = Path(__file__).parent.resolve()


def run_cart(timeout, depth, train_data, test_data, leaf_nodes=None, tune=True):
    X_train, y_train, train_info = load_data_cont_bincat(train_data)
    X_test, y_test, test_info = load_data_cont_bincat(test_data)

    if leaf_nodes is None:
        leaf_nodes = 2**depth

    parameters = {"max_depth": [depth], 'max_leaf_nodes': [leaf_nodes]}
    total_train_var = np.std(y_train) * np.std(y_train)

    start_time = time.time() # Start timer after reading data

    if tune:
        reg = DecisionTreeRegressor()
        parameters["ccp_alpha"] = (
            np.array([0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001])
            * total_train_var
        )

        tuning_model = GridSearchCV(
            reg, param_grid=parameters, scoring="neg_mean_squared_error", cv=5, verbose=0
        )
        tuning_model.fit(X_train, y_train)
        reg = DecisionTreeRegressor(**tuning_model.best_params_)
    else:
        reg = DecisionTreeRegressor(max_depth=depth, max_leaf_nodes=leaf_nodes)
        
    reg.fit(X_train, y_train)

    duration = time.time() - start_time

    return {
        "time": duration,
        "train_r2": r2_score(y_train, reg.predict(X_train)),
        "test_r2": r2_score(y_test, reg.predict(X_test)),
        "leaves": reg.get_n_leaves(),
        "terminal_calls": -1,
    }
