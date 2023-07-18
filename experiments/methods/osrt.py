from pathlib import Path
from methods.misc.tree_classifier import TreeClassifier
import json
import re
import tempfile
import pandas as pd
import numpy as np
import subprocess
import sys
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).parent.resolve()

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"  # https://docs.python.org/3/library/re.html#simulating-scanf


def compute_mse(model, X, y, loss_normalizer):
    return mean_squared_error(y, model.predict(X) * loss_normalizer)


def parse_output(content, timeout: int, model_output_path, train_data, test_data):
    props = {}
    if "Training Duration: " not in content:
        # Timeout
        props["time"] = timeout + 1
        props["train_mse"] = -1
        props["test_mse"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    time_pattern = r"Training Duration: (" + float_pattern + ") seconds"
    loss_normalizer_pattern = r"loss_normalizer: (" + float_pattern + ")"
    props["time"] = float(re.search(time_pattern, content, re.M).group(1))
    props["terminal_calls"] = -1

    df_train = pd.read_csv(train_data)
    X_train = df_train[df_train.columns[:-1]].to_numpy()
    y_train = df_train[df_train.columns[-1]].to_numpy()

    df_test = pd.read_csv(test_data)
    X_test = df_test[df_test.columns[:-1]].to_numpy()
    y_test = df_test[df_test.columns[-1]].to_numpy()

    # OSRT reports False-convergence detected when a single root node is the best. Special case for this here
    if re.search("False-convergence Detected", content):
        props["leaves"] = 1
        props["train_mse"] = mean_squared_error(
            y_train, np.full(len(y_train), np.mean(y_train))
        )
        props["test_mse"] = mean_squared_error(
            y_test, np.full(len(y_test), np.mean(y_train))
        )
    else:
        loss_normalizer = float(
            re.search(loss_normalizer_pattern, content, re.M).group(1)
        )
        with open(model_output_path) as f:
            models = json.load(f)
        classifier = TreeClassifier(models[0])
        props["train_mse"] = compute_mse(classifier, X_train, y_train, loss_normalizer)
        props["test_mse"] = compute_mse(classifier, X_test, y_test, loss_normalizer)
        props["leaves"] = classifier.leaves()
    return props


def run_osrt(exe, timeout, depth, train_data, test_data, cp, tune):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        model_output_path = dir_path / "model.json"
        config_path = dir_path / "osrt_config.json"
        with open(SCRIPT_DIR / "misc" / "osrt_config.json") as config_file:
            config_base = json.load(config_file)
            config_base["depth_budget"] = (
                depth + 1
            )  # OSRT considers root with 2 leaves as depth 2, while STreeD considers it depth 1
            config_base["regularization"] = cp
            config_base["model"] = str(model_output_path)
            with open(config_path, "w") as tmp_config_file:
                json.dump(config_base, tmp_config_file)

        try:
            result = subprocess.check_output(
                [
                    "timeout",
                    str(timeout),
                    exe,
                    train_data,
                    config_path,
                ],
                timeout=timeout,
            )
            output = result.decode()
            # print(output)
            parsed = parse_output(
                output, timeout, model_output_path, train_data, test_data
            )
            return parsed
        except subprocess.TimeoutExpired as e:
            # print(e.stdout.decode())
            return parse_output("", timeout)
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr, flush=True)
            return {
                "time": -1,
                "train_mse": -1,
                "test_mse": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }
