import os
import subprocess
import json
import re

# Parts of this code were taken from https://github.com/ruizhang1996/regression-tree-benchmark

def run_osrt(dataset, depth, timeout=30.0):
    txt_pattern = {
        "loss_normalizer": (r"loss_normalizer: ([\d.]+)", float),
        "loss": (r"Loss: ([\d.]+)", float),
        "complexity": (r"Complexity: ([\d.]+)", float),
        "time": (r"Training Duration: ([\d.]+) seconds", float),
    }

    def parse_output(output):
        out = {}
        for i in txt_pattern:
            out[i] = txt_pattern[i][1](
                re.search(txt_pattern[i][0], output, re.M).group(1)
            )
        return out

    with open("./osrt_config.json") as config_file:
        config_base = json.load(config_file)
        config_base["depth_budget"] = depth
        with open("./osrt_config.tmp.json", "w") as tmp_config_file:
            json.dump(config_base, tmp_config_file)

    osrt_base = "./data/osrt"
    csv_path = os.path.join(osrt_base, dataset)

    msys_env = os.environ.copy()
    msys_env["PATH"] = "C:\\msys64\\ucrt64\\bin\\;" + msys_env["PATH"]
    try:
        result = subprocess.check_output(
            ["../../optimal-sparse-regression-tree-public/gosdt",
             csv_path, "./osrt_config.tmp.json"], env=msys_env, timeout=timeout)
        output = result.decode()
        parsed = parse_output(output)
        parsed["stdout"] = output
        return parsed
    except subprocess.TimeoutExpired as e:
        return {"time": "Expired", "stdout": e.stdout.decode()}
    except subprocess.CalledProcessError as e:
        return {"time": "Crashed", "stdout": e.stdout.decode()}


def run_streed(dataset, depth, timeout=30.0):
    txt_pattern = {
        "time": (r"CLOCKS FOR SOLVE: ([\d.]+)", float),
    }

    def parse_output(output):
        out = {}
        for i in txt_pattern:
            out[i] = txt_pattern[i][1](
                re.search(txt_pattern[i][0], output, re.M).group(1)
            )
        return out
    
    streed_base = "./data/streed"
    csv_path = os.path.join(streed_base, dataset)
    try:
        result = subprocess.check_output(
            ["../../streed2/out/build/x64-Release/STREED",
             "-task", "regression",
             "-file", csv_path,
             "-max-depth", str(depth),
             "-max-num-nodes", str(2**depth - 1)], timeout=timeout)
        output = result.decode()
        parsed = parse_output(output)
        parsed["stdout"] = output
        return parsed
    except subprocess.TimeoutExpired as e:
        return {"time": "Expired", "stdout": e.stdout.decode()}
    except subprocess.CalledProcessError as e:
        return {"time": "Crashed", "stdout": e.stdout.decode()}
    

def run(dataset, depth):
    print(f"Running OSRT benchmark for {dataset} with max depth of {depth}")
    osrt_out = run_osrt(dataset, depth)
    print(f"Running STreeD benchmark for {dataset} with max depth of {depth}")
    streed_out = run_streed(dataset, depth)

    return {
        "osrt": osrt_out,
        "streed": streed_out
    }

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
    for dataset in dataset_files[:2]:
        dataset_results = []
        for depth in range(2, 8):
            result = {
                "depth": depth,
                "experiment": run(dataset, depth)
            }
            dataset_results.append(result)
        results.append({
            "dataset": dataset,
            "results": dataset_results
        })
    
    with open("./results/experiment.json", "w") as experiment_output:
        json.dump(results, experiment_output, indent=4)