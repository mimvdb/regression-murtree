from misc.tree_classifier import TreeClassifier
import json
import re
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?" # https://docs.python.org/3/library/re.html#simulating-scanf

def compute_mse(model, X, y, loss_normalizer):
    return mean_squared_error(y, model.predict(X) * loss_normalizer)

def parse_output(content, timeout: int, csv_path, model_output_path):
    props = {}
    if "Training Duration: " not in content:
        # Timeout
        props["time"] = timeout + 1
        props["train_mse"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    time_pattern = r"Training Duration: (" + float_pattern + ") seconds"
    loss_normalizer_pattern = r"loss_normalizer: (" + float_pattern + ")"
    props["time"] = float(re.search(time_pattern, content, re.M).group(1))
    props["terminal_calls"] = 0

    df = pd.read_csv(csv_path)
    X_train = df[df.columns[:-1]].to_numpy()
    y_train = df[df.columns[-1]].to_numpy()
    
    # OSRT reports False-convergence detected when a single root node is the best. Special case for this here
    if re.search("False-convergence Detected", content):
        props["leaves"] = 1
        props["train_mse"] = mean_squared_error(y_train, np.full(len(y_train), np.mean(y_train)))
    else:
        loss_normalizer = float(re.search(loss_normalizer_pattern, content, re.M).group(1))
        with open(model_output_path) as f:
            models = json.load(f)
        classifier = TreeClassifier(models[0])
        props["train_mse"] = compute_mse(classifier, X_train, y_train, loss_normalizer)
        props["leaves"] = classifier.leaves()
    return props