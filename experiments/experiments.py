import os
import subprocess

def run_osrt(dataset, timeout=30.0):
    osrt_base = "./data/osrt"
    csv_path = os.path.join(osrt_base, dataset)

    msys_env = os.environ.copy()
    msys_env["PATH"] = "C:\\msys64\\ucrt64\\bin\\;" + msys_env["PATH"]
    try:
        result = subprocess.run(["../../optimal-sparse-regression-tree-public/gosdt", csv_path, "./osrt_config.json"], env=msys_env, capture_output=True, timeout=timeout)
        return result.stdout.decode()
    except subprocess.TimeoutExpired:
        return "expired"

def run_streed(dataset, timeout=30.0):
    streed_base = "./data/streed"
    csv_path = os.path.join(streed_base, dataset)
    try:
        result = subprocess.run(["../../streed2/out/build/x64-Release/STREED", "-task", "regression", "-file", csv_path], capture_output=True, timeout=timeout)
        return result.stdout.decode()
    except subprocess.TimeoutExpired:
        return "expired"
    

def run(dataset):
    print(f"Running benchmark for {dataset}")

    print(run_osrt(dataset))
    print(run_streed(dataset))

if __name__ == "__main__":
    datasets_path = "./data/datasets.txt"
    if not os.path.exists(datasets_path):
        print("./data/datasets.txt not found. Run ./prepare_data.py to generate datasets\n")
        exit()
    
    dataset_files = []
    with open(datasets_path) as datasets_file:
        dataset_files.extend([f.strip() for f in datasets_file.readlines()])

    run(dataset_files[0])