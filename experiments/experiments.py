from model.tree_classifier import TreeClassifier
import csv
import os
import subprocess
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Parts of this code were taken from https://github.com/ruizhang1996/regression-tree-benchmark
timed_out_osrt = set()
timed_out_streed = set()

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"# https://docs.python.org/3/library/re.html#simulating-scanf

def compute_mse(model, X, y, loss_normalizer):
    return mean_squared_error(y, model.predict(X) * loss_normalizer)

def run_osrt(dataset, depth, cost_complexity, timeout):
    timeout_result = {
        "time": timeout + 1,
        "train_mse": -1,
        "leaves": -1,
        "terminal_calls": -1
    }
    if ((dataset, depth - 1, cost_complexity) in timed_out_osrt):
        timed_out_osrt.add((dataset, depth, cost_complexity))
        return timeout_result

    model_output_path = "./osrt_model.json"
    osrt_base = "./data/osrt"
    csv_path = os.path.join(osrt_base, dataset)

    def parse_output(output):
        out = {}
        time_pattern = r"Training Duration: (" + float_pattern + ") seconds"
        loss_normalizer_pattern = r"loss_normalizer: (" + float_pattern + ")"
        out["time"] = float(re.search(time_pattern, output, re.M).group(1))
        out["terminal_calls"] = 0
        # OSRT reports False-convergence detected when a single root node is the best. Special case for this here
        df = pd.read_csv(csv_path)
        X_train = df[df.columns[:-1]].to_numpy()
        y_train = df[df.columns[-1]].to_numpy()
        if re.search("False-convergence Detected", output):
            out["leaves"] = 1
            out["train_mse"] = mean_squared_error(y_train, np.mean(y_train))
        else:
            loss_normalizer = float(re.search(loss_normalizer_pattern, output, re.M).group(1))
            with open(model_output_path) as f:
                models = json.load(f)
            classifier = TreeClassifier(models[0])
            out["train_mse"] = compute_mse(classifier, X_train, y_train, loss_normalizer)
            out["leaves"] = classifier.leaves()
        return out

    with open("./osrt_config.json") as config_file:
        config_base = json.load(config_file)
        config_base["depth_budget"] = depth + 1 # OSRT root with 2 leaves as depth 2, while STreeD considers it depth 1
        config_base["regularization"] = cost_complexity
        config_base["model"] = model_output_path
        with open("./osrt_config.tmp.json", "w") as tmp_config_file:
            json.dump(config_base, tmp_config_file)

    msys_env = os.environ.copy()
    msys_env["PATH"] = "C:\\msys64\\ucrt64\\bin\\;" + msys_env["PATH"]
    try:
        result = subprocess.check_output(
            ["../../optimal-sparse-regression-tree-public/gosdt",
             csv_path, "./osrt_config.tmp.json"], env=msys_env, timeout=timeout)
        output = result.decode()
        #print(output)
        parsed = parse_output(output)
        return parsed
    except subprocess.TimeoutExpired as e:
        print(e.stdout.decode())
        timed_out_osrt.add((dataset, depth, cost_complexity))
        return timeout_result
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode())
        return {
            "time": -1,
            "train_mse": -1,
            "leaves": -1,
            "terminal_calls": -1
        }


def run_streed(dataset, depth, cost_complexity, timeout, use_lower, use_custom, use_kmeans):
    timeout_result = {
        "time": timeout + 1,
        "train_mse": -1,
        "leaves": -1,
        "terminal_calls": -1
    }
    if ((dataset, depth - 1, cost_complexity, use_lower, use_custom, use_kmeans) in timed_out_streed):
        timed_out_streed.add((dataset, depth, cost_complexity, use_lower, use_custom, use_kmeans))
        return timeout_result

    txt_pattern = {
        "terminal_calls": (r"Terminal calls: (\d+)", int),
        "train_mse": (r"Solution 0:\s+\d+\s+\d+\s+(" + float_pattern + ")", float),
        "leaves": (r"Solution 0:\s+\d+\s+(\d+)", int),
        "time": (r"CLOCKS FOR SOLVE: (" + float_pattern + ")", float),
    }

    def parse_output(output):
        out = {}
        for i in txt_pattern:
            out[i] = txt_pattern[i][1](
                re.search(txt_pattern[i][0], output, re.M).group(1)
            )
        out["leaves"] += 1 # Solution prints branching nodes, but want leaf nodes
        return out
    
    streed_base = "./data/streed"
    csv_path = os.path.join(streed_base, dataset)
    try:
        result = subprocess.check_output(
            ["../../streed-regression/out/build/x64-Release/STREED",
             "-task", "cost-complex-regression",
             "-file", csv_path,
             "-max-depth", str(depth),
             "-max-num-nodes", str(2**depth - 1),
             "-time", str(timeout + 10),
             "-use-lower-bound", "1" if use_lower else "0",
             "-use-task-lower-bound", "1" if use_custom else "0",
             "-regression-bound", "kmeans" if use_kmeans else "equivalent",
             "-cost-complexity", str(cost_complexity)], timeout=timeout)
        output = result.decode()
        #print(output)
        parsed = parse_output(output)
        return parsed
    except subprocess.TimeoutExpired as e:
        print(e.stdout.decode())
        timed_out_streed.add((dataset, depth, cost_complexity, use_lower, use_custom, use_kmeans))
        return timeout_result
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode())
        return {
            "time": -1,
            "train_mse": -1,
            "leaves": -1,
            "terminal_calls": -1
        }
    

def write_row(path, row):
    with open(path, "a", newline="") as osrt_file:
        writer = csv.DictWriter(osrt_file, fieldnames=[
            "dataset", "depth", "cost_complexity", "time", "train_mse", "leaves", "terminal_calls", "timeout"])
        if (osrt_file.tell() == 0):
            writer.writeheader()
        writer.writerow(row)

def enhance(obj, dataset, depth, cost_complexity, timeout):
    obj["depth"] = depth
    obj["dataset"] = dataset
    obj["cost_complexity"] = cost_complexity
    obj["timeout"] = timeout

def run(csv_base, dataset, depth, cost_complexity, timeout=30):
    print(f"Running STreeD benchmark for {dataset} with max depth of {depth} and cost complexity {cost_complexity}")
    streed_out = run_streed(dataset, depth, cost_complexity, timeout, True, True, True)
    enhance(streed_out, dataset, depth, cost_complexity, timeout)
    write_row(f"{csv_base}/{dataset[:-4]}/streed_all.csv", streed_out)

    print(f"Running STreeD benchmark for {dataset} with max depth of {depth} and cost complexity {cost_complexity}")
    streed_out = run_streed(dataset, depth, cost_complexity, timeout, False, False, False)
    enhance(streed_out, dataset, depth, cost_complexity, timeout)
    write_row(f"{csv_base}/{dataset[:-4]}/streed_none.csv", streed_out)

    print(f"Running STreeD benchmark for {dataset} with max depth of {depth} and cost complexity {cost_complexity}")
    streed_out = run_streed(dataset, depth, cost_complexity, timeout, True, False, False)
    enhance(streed_out, dataset, depth, cost_complexity, timeout)
    write_row(f"{csv_base}/{dataset[:-4]}/streed_similarity.csv", streed_out)

    print(f"Running STreeD benchmark for {dataset} with max depth of {depth} and cost complexity {cost_complexity}")
    streed_out = run_streed(dataset, depth, cost_complexity, timeout, True, True, False)
    enhance(streed_out, dataset, depth, cost_complexity, timeout)
    write_row(f"{csv_base}/{dataset[:-4]}/streed_equivalent.csv", streed_out)

    print(f"Running OSRT benchmark for {dataset} with max depth of {depth} and cost complexity {cost_complexity}")
    osrt_out = run_osrt(dataset, depth, cost_complexity, timeout)
    enhance(osrt_out, dataset, depth, cost_complexity, timeout)
    write_row(f"{csv_base}/{dataset[:-4]}/osrt.csv", osrt_out)


if __name__ == "__main__":
    datasets_path = "./data/datasets.txt"
    if not os.path.exists(datasets_path):
        print("./data/datasets.txt not found. Run ./prepare_data.py to generate datasets\n")
        exit()
    
    dataset_files = []
    with open(datasets_path) as datasets_file:
        dataset_files.extend([f.strip() for f in datasets_file.readlines()])

    os.makedirs("./results", exist_ok=True)

    results = []
    complexities = list(np.concatenate([[0.0001, 0.0002, 0.0005], np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.11, 0.025), [0.1, 0.2, 0.5]]))
    for dataset in dataset_files:
        os.makedirs(f"./results/{dataset[:-4]}", exist_ok=False)
        for depth in range(2, 11):
            for cost_complexity in complexities:
                run("./results", dataset, depth, cost_complexity)
