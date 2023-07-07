#! /usr/bin/env python

from pathlib import Path
import csv
import json
import logging
import os
import numpy as np

from lab.environments import LocalEnvironment
from lab.environments import SlurmEnvironment
from lab.experiment import Experiment

from lab.reports import Report
from lab import tools

SCRIPT_DIR = Path(__file__).parent.resolve()
OSRT_PATH = SCRIPT_DIR / ".." / ".." / "optimal-sparse-regression-tree-public" / "build" / "gosdt" # SCRIPT_DIR / "gosdt"
STREED_PATH = SCRIPT_DIR / ".." / ".." / "streed2" / "build" / "STREED" # SCRIPT_DIR / "STREED"

class CsvReport(Report):
    def __init__(self, attributes=None, filter=None, delimiter=',', **kwargs):
        self.attributes = tools.make_list(attributes)
        self.output_format = format
        self.toc = True
        self.run_filter = tools.RunFilter(filter, **kwargs)

        self.delimiter = delimiter


    def write(self):
        with open(self.outfile, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)

            writer.writerow(self.attributes)

            for run in self.props.values():
                row = [get_attribute_value(attribute, run) for attribute in self.attributes]
                writer.writerow(row)

        logging.info(f"Wrote file://{self.outfile}")


def get_attribute_value(attribute: str, run_props: dict) -> str:
    if attribute in run_props:
        return str(run_props[attribute])

    return ""

class DelftBlueEnvironment(SlurmEnvironment):
    MAX_TASKS = 1_000

    def __init__(
        self,
        email=None,
        account="innovation",
        partition="compute",
        time_limit_per_task=None,
        memory_per_cpu=None
    ):
        super().__init__(
            email=email, 
            extra_options=f"#SBATCH --account={account}", 
            partition=partition, 
            time_limit_per_task=time_limit_per_task, 
            memory_per_cpu=memory_per_cpu,
            qos="normal")

def add_osrt_run(experiment: Experiment, timeout: int, dataset: str, depth: int, cost_complexity: float, sequence: int):
    run = experiment.add_run()

    id_str = f"osrt_{dataset}_{depth}_{cost_complexity}_{sequence}"

    model_output_path = SCRIPT_DIR / "tmp" / "osrt" / "models" / f"{id_str}.json"
    config_path = SCRIPT_DIR / "tmp" / "osrt" / "configs" / f"{id_str}.json"
    dataset_path = SCRIPT_DIR / "data_large" / "osrt" / dataset
    os.makedirs(model_output_path.parent.resolve(), exist_ok=True)
    os.makedirs(config_path.parent.resolve(), exist_ok=True)
    with open(SCRIPT_DIR / "osrt_config.json") as config_file:
        config_base = json.load(config_file)
        config_base["depth_budget"] = depth + 1 # OSRT considers root with 2 leaves as depth 2, while STreeD considers it depth 1
        config_base["regularization"] = cost_complexity
        config_base["model"] = str(model_output_path)
        with open(config_path, "w") as tmp_config_file:
            json.dump(config_base, tmp_config_file)

    run.add_command(f"osrt", ["timeout", timeout, OSRT_PATH, dataset_path, config_path])

    # Set unique id
    run.set_property("id", ["osrt", dataset, str(depth), str(cost_complexity), str(sequence)])
    run.set_property("id_str", id_str)
    run.set_property("timeout", timeout)
    run.set_property("algorithm", "osrt")
    run.set_property("dataset", dataset)
    run.set_property("depth", depth)
    run.set_property("cost_complexity", cost_complexity)
    run.set_property("sequence", sequence)
    run.set_property("model_output_path", str(model_output_path))
    run.set_property("csv_path", str(dataset_path))

def add_streed_run(experiment: Experiment, timeout: int, dataset: str, depth: int, cost_complexity: float, sequence: int, extra_label: str, use_lower: bool, use_custom: bool, use_kmeans: bool):
    run = experiment.add_run()

    id_str = f"streed_{extra_label}_{dataset}_{depth}_{cost_complexity}_{sequence}_{extra_label}"

    dataset_path = SCRIPT_DIR / "data_large" / "streed" / dataset

    run.add_command(f"streed_{extra_label}",
                    ["timeout", timeout, STREED_PATH,
                    "-task", "cost-complex-regression",
                    "-file", dataset_path,
                    "-max-depth", str(depth),
                    "-max-num-nodes", str(2**depth - 1),
                    "-time", str(timeout + 10),
                    "-use-lower-bound", "1" if use_lower else "0",
                    "-use-task-lower-bound", "1" if use_custom else "0",
                    "-regression-bound", "kmeans" if use_kmeans else "equivalent",
                    "-cost-complexity", str(cost_complexity)])

    # Set unique id
    run.set_property("id", ["streed", extra_label, dataset, str(depth), str(cost_complexity), str(sequence)])
    run.set_property("id_str", id_str)
    run.set_property("timeout", timeout)
    run.set_property("algorithm", f"streed_{extra_label}")
    run.set_property("dataset", dataset)
    run.set_property("depth", depth)
    run.set_property("cost_complexity", cost_complexity)
    run.set_property("sequence", sequence)
    run.set_property("csv_path", str(dataset_path))

def add_runs(experiment: Experiment, timeout: int):
    datasets_path = SCRIPT_DIR / "data" / "datasets_large.txt"
    if not os.path.exists(datasets_path):
        print("datasets.txt not found. Run ./prepare_data.py to generate datasets\n")
        exit()
    
    dataset_files = []
    with open(datasets_path) as datasets_file:
        dataset_files.extend([f.strip() for f in datasets_file.readlines()])

    for dataset in dataset_files:
        for i in range(3):
            depth = 5
            cost_complexity = 0.035
            add_osrt_run(experiment, timeout, dataset, depth, cost_complexity, i)
            add_streed_run(experiment, timeout, dataset, depth, cost_complexity, i, "none", False, False, False)
            add_streed_run(experiment, timeout, dataset, depth, cost_complexity, i, "similarity", True, False, False)
            add_streed_run(experiment, timeout, dataset, depth, cost_complexity, i, "equivalent", True, True, False)
            add_streed_run(experiment, timeout, dataset, depth, cost_complexity, i, "all", True, True, True)


if __name__ == "__main__":

    env =\
        LocalEnvironment(processes=1) if False else\
        DelftBlueEnvironment(
            "",
            "",
            "compute",
            "00:45:00",
            "100G"
        )

    # The folder in which experiment files are generated.
    exp = Experiment(environment=env, path=SCRIPT_DIR / "experiment_scalabilty")
    exp.add_parser("parser.py")

    add_runs(exp, 30*60)

    exp.add_step("build", exp.build)
    exp.add_step("start", exp.start_runs)
    exp.add_fetcher(name="fetch")
    # exp.add_parse_again_step()

    exp.add_report(CsvReport(attributes=[
        "timeout",
        "algorithm",
        "dataset",
        "depth",
        "cost_complexity",
        "sequence",
        "leaves",
        "train_mse",
        "time",
        "terminal_calls"
    ]), outfile="report.csv")

    exp.run_steps()
