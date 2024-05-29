from pathlib import Path
from methods.misc.tree_classifier import TreeClassifier
from methods.misc.util import load_data_bin_bincat
import json
import re
import tempfile
import pandas as pd
import numpy as np
import subprocess
import sys
import os
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).parent.resolve()
PREFIX_DATA = SCRIPT_DIR / ".." / ".." / "data" / "prepared" / "osrt"

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"  # https://docs.python.org/3/library/re.html#simulating-scanf


def compute_r2(model, X, y, loss_normalizer):
    return r2_score(y, model.predict(X.to_numpy()) * loss_normalizer)


def parse_output(content, timeout: int, model_output_path, train_data, test_data):
    props = {}

    loss_normalizer_pattern = r"loss_normalizer: (" + float_pattern + ")"

    if "Training Duration: " not in content:
        # Timeout
        train_r2 = -1
        if ("loss_normalizer:" in content) and ("Objective: [" in content):
            training_score_patern = r"Objective: \["+ float_pattern+", "+ float_pattern+"\]"
            all_matches = re.finditer(training_score_patern, content)
            for match in all_matches:
                value = float(match.group(4))
                if 1 - value > train_r2:
                    train_r2 = 1 - value # note that this value includes the CP costs
        
        props["time"] = timeout + 1
        props["train_r2"] = train_r2
        props["test_r2"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    time_pattern = r"Training Duration: (" + float_pattern + ") seconds"
    props["time"] = float(re.search(time_pattern, content, re.M).group(1))
    props["terminal_calls"] = -1

    if train_data.endswith(".csv"): # Hypertuning subrun
        train_df = pd.read_csv(train_data)
        X_train = train_df[train_df.columns[:-1]]
        y_train = train_df[train_df.columns[-1]]
        test_df = pd.read_csv(test_data)
        X_test = test_df[test_df.columns[:-1]]
        y_test = test_df[test_df.columns[-1]]
    else:
        X_train, y_train, train_info = load_data_bin_bincat(train_data)
        X_test, y_test, test_info = load_data_bin_bincat(test_data)

    # OSRT reports False-convergence detected when a single root node is the best. Special case for this here
    if re.search("False-convergence Detected", content):
        props["leaves"] = 1
        props["train_r2"] = r2_score(
            y_train, np.full(len(y_train), np.mean(y_train))
        )
        props["test_r2"] = r2_score(
            y_test, np.full(len(y_test), np.mean(y_train))
        )
    else:
        loss_normalizer = float(
            re.search(loss_normalizer_pattern, content, re.M).group(1)
        )
        with open(model_output_path, encoding="utf-8") as f:
            models = json.load(f)
        classifier = TreeClassifier(models[0])
        props["train_r2"] = compute_r2(classifier, X_train, y_train, loss_normalizer)
        props["test_r2"] = compute_r2(classifier, X_test, y_test, loss_normalizer)
        props["leaves"] = classifier.leaves()
    return props


def run_osrt(exe, timeout, depth, train_data, test_data, cp, tune):
    if tune: return _hyper(exe, timeout, depth, train_data, test_data)
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
            config_base["time_limit"] == str(timeout)
            with open(config_path, "w") as tmp_config_file:
                json.dump(config_base, tmp_config_file)

        try:
            train_file_path = str(PREFIX_DATA / (train_data + ".csv")) if not train_data.endswith(".csv") else train_data
            command = [
                    exe,
                    train_file_path,
                    config_path,
                ]
            
            if os.name != "nt": 
                command = ["timeout", str(timeout + 20)] + command

            result = subprocess.check_output(command,timeout=timeout)
            output = result.decode()
            #print(output)
            parsed = parse_output(
                output, timeout, model_output_path, train_data, test_data
            )
            return parsed
        except subprocess.TimeoutExpired as e:
            #print(e.stdout.decode())
            return parse_output(e.stdout.decode(), timeout, "", "", "")
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr, flush=True)
            return {
                "time": -1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }

def _hyper(exe, timeout, depth, train_data, test_data):
    
    start = time.time()

    df = pd.read_csv(str(PREFIX_DATA / (train_data + ".csv")))
    label = df[df.columns[-1]] # last column is the label
    # use label to stratify the data
    discretized_label = KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile").fit_transform(np.array(label).reshape(-1,1))

    configs = [0.1, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0005, 0.0001]
    scores_per_config = [0] * len(configs)

    for run in range(5):
        train_df, test_df = train_test_split(df, test_size = 0.2, random_state=42+run, stratify=discretized_label)
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            train_path = dir_path / f"{train_data}-'val-train-{run+1}.csv"
            test_path = dir_path / f"{test_data}-'val-test-{run+1}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            for i, cp in enumerate(configs):
                if time.time() - start >= timeout: break
                run = run_osrt(exe, timeout - (time.time() - start), depth, str(train_path), str(test_path), cp, False)
                scores_per_config[i] += run["test_r2"]

        if time.time() - start >= timeout: break
    
    if time.time() - start >= timeout: 
        return parse_output("", timeout, "", "", "")

    best_config = np.argmax(scores_per_config)
    result = run_osrt(exe, timeout - (time.time() - start), depth, train_data, test_data, configs[best_config], False)
    if result["time"] == -1 or result["time"] == timeout + 1:
        return result
    result["time"] = time.time() - start
    return result