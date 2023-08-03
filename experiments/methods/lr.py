from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from methods.misc.util import load_data_cont_bincat

SCRIPT_DIR = Path(__file__).parent.resolve()


def run_lr(train_data, test_data):
    X_train, y_train, train_info = load_data_cont_bincat(train_data)
    X_test, y_test, test_info = load_data_cont_bincat(test_data)

    parameters={"l1_ratio": [1.0], "max_iter": [10000]}

    start_time = time.time() # Start timer after reading data
    
    reg = ElasticNet()
    parameters["alpha"] = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
    tuning_model=GridSearchCV(reg, param_grid=parameters, scoring='neg_mean_squared_error',cv=5,verbose=0)
    tuning_model.fit(X_train, y_train)
    reg = ElasticNet(**tuning_model.best_params_)
    reg.fit(X_train, y_train)

    duration = time.time() - start_time

    return {
        "time": duration,
        "train_r2": r2_score(y_train, reg.predict(X_train)),
        "test_r2": r2_score(y_test, reg.predict(X_test)),
        "leaves": 1,
        "terminal_calls": -1,
    }
