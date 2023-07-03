from model.tree_classifier import TreeClassifier
import csv
import os
import subprocess
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    result = subprocess.check_output(
        ["../../streed2/build/STREED",
            "-task", "cost-complex-regression",
            "-file", "./data/streed/insurance.csv",
            "-max-depth", "4",
            "-max-num-nodes", "15",
            "-time", "20",
            "-use-lower-bound", "1",
            "-use-task-lower-bound", "1",
            "-regression-bound", "kmeans",
            "-cost-complexity", "0.085"])
    output = result.decode()
    print(output)
except subprocess.CalledProcessError as e:
    print(e.stdout.decode())