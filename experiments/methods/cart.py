import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def run_cart(timeout, depth, train_data, test_data):
    train_df = pd.read_csv(train_data, sep=" ", header=None)
    test_df = pd.read_csv(test_data, sep=" ", header=None)

    X_train = train_df.iloc[:, 1:]
    y_train = train_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]

    parameters = {"max_depth": [depth]}
    total_train_var = np.std(y_train) * np.std(y_train)

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
    reg.fit(X_train, y_train)
    return {
        "time": -1,
        "train_mse": -1,
        "test_mse": -1,
        "leaves": -1,
        "terminal_calls": -1,
    }
