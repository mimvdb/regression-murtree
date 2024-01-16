from pathlib import Path
from methods.misc.util import load_data_info
import re
import tempfile
import subprocess
import sys
import os

SCRIPT_DIR = Path(__file__).parent.resolve()
PREFIX_DATA = SCRIPT_DIR / ".." / ".." / "data" / "prepared" / "all"
GUIDE_CONFIG = SCRIPT_DIR / "misc" / "guide.in"
GUIDE_L_CONFIG = SCRIPT_DIR / "misc" / "guide_linreg.in"
GUIDE_SL_CONFIG = SCRIPT_DIR / "misc" / "guide_simple_linreg.in"

float_pattern = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"  # https://docs.python.org/3/library/re.html#simulating-scanf


def modify_script(original_lines, data_path):
    modified_lines = []
    for line in original_lines:
        if line.startswith("newdata <-"):
            modified_lines.append(f"newdata <- read.csv(\"{data_path}\",header=FALSE,colClasses=\"character\")\n")
        else:
            modified_lines.append(line)
    modified_lines.append("mse <- mean((as.numeric(newdata$label) - pred)^2)\n")
    modified_lines.append("cat(mse)\n")
    return modified_lines

def parse_output(content, timeout: int, model_output_path, train_data, test_data):
    props = {}
    if "Elapsed time in seconds" not in content:
        # Timeout
        print(content)
        props["time"] = timeout + 1
        props["train_r2"] = -1
        props["test_r2"] = -1
        props["leaves"] = -1
        props["terminal_calls"] = -1
        return props

    time_pattern = r"Elapsed time in seconds: (" + float_pattern + ")"
    leaves_pattern = r"Number of terminal nodes of final tree: (\d+)"
    props["time"] = float(re.search(time_pattern, content, re.M).group(1))
    props["leaves"] = int(re.search(leaves_pattern, content, re.M).group(1))
    props["terminal_calls"] = -1

    with open(model_output_path / "guide_predict.R", "r") as predict_file:
        predict_lines = predict_file.readlines()
    with open(model_output_path / "guide_predict_train.R", "w") as predict_file:
        lines = modify_script(predict_lines, str(PREFIX_DATA / (train_data + '.csv')))
        predict_file.writelines(lines)
    with open(model_output_path / "guide_predict_test.R", "w") as predict_file:
        lines = modify_script(predict_lines, str(PREFIX_DATA / (test_data + '.csv')))
        predict_file.writelines(lines)
    
    train_mse = float(subprocess.check_output(["Rscript", str(model_output_path / "guide_predict_train.R")]))
    test_mse = float(subprocess.check_output(["Rscript", str(model_output_path / "guide_predict_test.R")]))

    train_info = load_data_info(train_data)
    test_info = load_data_info(test_data)

    props["train_r2"] = 1 - (train_mse / train_info["mean_squared_error"])
    props["test_r2"] = 1 - (test_mse / test_info["mean_squared_error"])
    return props


def run_guide(exe, timeout, depth, train_data, test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        with open(GUIDE_CONFIG, "r") as template_file:
            template_lines = template_file.readlines()
        
        input_string = ""
        for line in template_lines:
            if "guide.in" in line:
                # 100 maximum character limit for description files, so symlink to it instead of full absolute path
                files_to_link = [train_data + ".guide.in", train_data + ".csv"]
                for file in files_to_link:
                    to_link = PREFIX_DATA / file
                    link = dir_path / file
                    os.symlink(to_link, link)
                input_string += line.replace("guide.in", str(train_data + ".guide.in"))
            elif "max. no. split levels" in line:
                input_string += line.replace("2", str(depth))
            else:
                input_string += line
        
        try:
            command = [exe]
            
            if os.name != "nt": 
                command = ["timeout", str(timeout)] + command

            result = subprocess.check_output(command, input=bytes(input_string,"utf-8"),timeout=timeout,cwd=str(dir_path))
            output = result.decode()
            parsed = parse_output(
                output, timeout, dir_path, train_data, test_data
            )
            return parsed
        except subprocess.TimeoutExpired as e:
            # print(e.stdout.decode())
            return {
                "time": timeout + 1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr, flush=True)
            return {
                "time": -1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }

def run_guide_sl(exe, timeout, depth, train_data, test_data):
    train_info = load_data_info(train_data)
    n_reg_cols = len(train_info["continuous_cols"])

    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        with open(GUIDE_SL_CONFIG, "r") as template_file:
            template_lines = template_file.readlines()
        
        input_string = ""
        for line in template_lines:
            if "guide.in" in line:
                # 100 maximum character limit for description files, so symlink to it instead of full absolute path
                files_to_link = [train_data + ".guide.in", train_data + ".csv"]
                for file in files_to_link:
                    to_link = PREFIX_DATA / file
                    link = dir_path / file
                    os.symlink(to_link, link)
                input_string += line.replace("guide.in", str(train_data + ".guide.in"))
            elif "max. no. split levels" in line:
                input_string += line.replace("2", str(depth))
            elif "min. node size" in line:
                input_string += line.replace("1", "2") + str(10) + "\n"
            else:
                input_string += line

        try:
            command = [exe]
            
            if os.name != "nt": 
                command = ["timeout", str(timeout)] + command

            result = subprocess.check_output(command, input=bytes(input_string,"utf-8"),timeout=timeout,cwd=str(dir_path))
            output = result.decode()
            parsed = parse_output(
                output, timeout, dir_path, train_data, test_data
            )
            return parsed
        except subprocess.TimeoutExpired as e:
            # print(e.stdout.decode())
            return {
                "time": timeout + 1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr, flush=True)
            return {
                "time": -1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }


def run_guide_l(exe, timeout, depth, train_data, test_data):
    train_info = load_data_info(train_data)
    n_reg_cols = len(train_info["continuous_cols"])

    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        with open(GUIDE_L_CONFIG, "r") as template_file:
            template_lines = template_file.readlines()
        
        input_string = ""
        for line in template_lines:
            if "guide.in" in line:
                # 100 maximum character limit for description files, so symlink to it instead of full absolute path
                files_to_link = [train_data + ".guide.in", train_data + ".csv"]
                for file in files_to_link:
                    to_link = PREFIX_DATA / file
                    link = dir_path / file
                    os.symlink(to_link, link)
                input_string += line.replace("guide.in", str(train_data + ".guide.in"))
            elif "max. no. split levels" in line:
                input_string += line.replace("2", str(depth))
            elif "min. node size" in line:
                input_string += line.replace("1", "2") + str(n_reg_cols * 10) + "\n"
            else:
                input_string += line

        try:
            command = [exe]
            
            if os.name != "nt": 
                command = ["timeout", str(timeout)] + command

            result = subprocess.check_output(command, input=bytes(input_string,"utf-8"),timeout=timeout,cwd=str(dir_path))
            output = result.decode()
            parsed = parse_output(
                output, timeout, dir_path, train_data, test_data
            )
            return parsed
        except subprocess.TimeoutExpired as e:
            # print(e.stdout.decode())
            return {
                "time": timeout + 1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(), file=sys.stderr, flush=True)
            return {
                "time": -1,
                "train_r2": -1,
                "test_r2": -1,
                "leaves": -1,
                "terminal_calls": -1,
            }
