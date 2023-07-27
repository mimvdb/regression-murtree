from pathlib import Path
from methods.misc.util import load_data_info
import re
import subprocess
import sys
import os
import json

SCRIPT_DIR = Path(__file__).parent.resolve()
PREFIX_DATA_PWC = SCRIPT_DIR / ".." / ".." / "data" / "prepared" / "streed_pwc"
PREFIX_DATA_PWL = SCRIPT_DIR / ".." / ".." / "data" / "prepared" / "streed_pwl"

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"  # https://docs.python.org/3/library/re.html#simulating-scanf


def parse_output(content, timeout, train_data, test_data):
    props = {}
    if "Solution 0" not in content:
        # Timeout
        props["time"] = timeout + 1
        props["train_r2"] = -1
        props["test_r2"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    # STreeD
    txt_pattern = {
        "terminal_calls": (r"Terminal calls: (\d+)", int),
        "solution": (r"Solution 0:\s+(.*)", str),
        "time": (r"CLOCKS FOR SOLVE: (" + float_pattern + ")", float),
    }

    matches = {}
    for i in txt_pattern:
        matches[i] = txt_pattern[i][1](
            re.search(txt_pattern[i][0], content, re.M).group(1)
        )

    # depth, branching nodes, train, test, avg. path length
    solution_vals = matches["solution"].split()

    props["terminal_calls"] = matches["terminal_calls"]
    props["time"] = matches["time"]
    props["leaves"] = (
        int(solution_vals[1]) + 1
    )  # Solution prints branching nodes, but want leaf nodes

    train_mse = float(solution_vals[2])
    test_mse = float(solution_vals[3])
    train_info = load_data_info(train_data)
    test_info = load_data_info(test_data)

    props["train_r2"] = 1 - (train_mse / train_info["mean_squared_error"])
    props["test_r2"] = 1 - (test_mse / test_info["mean_squared_error"])
    return props


def run_streed_pwc(
    exe,
    timeout,
    depth,
    train_data,
    test_data,
    cp,
    tune,
    use_kmeans,
    use_task_bound,
    use_lower_bound,
    use_d2,  # TODO: add CLI param to streed to toggle terminal solver
):
    try:
        command = [
            exe,
            "-task",
            "cost-complex-regression",
            "-mode",
            "hyper" if tune else "direct",
            "-file",
            str(PREFIX_DATA_PWC / (train_data + ".csv")),
            "-test-file",
            str(PREFIX_DATA_PWC / (test_data + ".csv")),
            "-max-depth",
            str(depth),
            "-max-num-nodes",
            str(2**depth - 1),
            "-time",
            str(timeout + 10),
            "-use-lower-bound",
            "1" if use_lower_bound else "0",
            "-use-task-lower-bound",
            "1" if use_task_bound else "0",
            "-regression-bound",
            "kmeans" if use_kmeans else "equivalent",
            "-cost-complexity",
            str(cp),
        ]

        # Add timeout, if not running on windows
        # (Windows timeout command is different)
        if os.name != "nt":
            command = ["timeout", str(timeout)] + command

        # print(" ".join(command))
        result = subprocess.check_output(command, timeout=timeout)
        output = result.decode()
        # print(output)
        parsed = parse_output(output, timeout, train_data, test_data)
        return parsed
    except subprocess.TimeoutExpired as e:
        # print(e.stdout.decode())
        return parse_output("", timeout, train_data, test_data)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode(), file=sys.stderr, flush=True)
        return {
            "time": -1,
            "train_r2": -1,
            "test_r2": -1,
            "leaves": -1,
            "terminal_calls": -1,
        }


def run_streed_pwl(exe, timeout, depth, train_data, test_data, cp, lasso, tune):
    with open(PREFIX_DATA_PWL / ".." / (train_data + ".json"), "r") as json_file:
        info = json.load(json_file)
        continuous_features = info["continuous_features"]

    try:
        command = [
            exe,
            "-task",
            "piecewise-linear-regression",
            "-mode",
            "hyper" if tune else "direct",
            "-file",
            str(PREFIX_DATA_PWL / (train_data + ".csv")),
            "-test-file",
            str(PREFIX_DATA_PWL / (test_data + ".csv")),
            "-max-depth",
            str(depth),
            "-max-num-nodes",
            str(2**depth - 1),
            "-time",
            str(timeout + 10),
            "-cost-complexity",
            str(cp),
            "-lasso-penalty",
            str(lasso),
            "-num-extra-cols",
            str(continuous_features),
            "-min-leaf-node-size",
            str(continuous_features)
        ]

        # Add timeout, if not running on windows
        # (Windows timeout command is different)
        if os.name != "nt":
            command = ["timeout", str(timeout)] + command

        print(" ".join(command))
        result = subprocess.check_output(command, timeout=timeout)
        output = result.decode()
        # print(output)
        parsed = parse_output(output, timeout, train_data, test_data)
        return parsed
    except subprocess.TimeoutExpired as e:
        # print(e.stdout.decode())
        return parse_output("", timeout, train_data, test_data)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode(), file=sys.stderr, flush=True)
        return {
            "time": -1,
            "train_r2": -1,
            "test_r2": -1,
            "leaves": -1,
            "terminal_calls": -1,
        }
