# Regression MurTree
This repository contains scripts to run experiments comparing the runtime of regression tree training algorithms

# Prerequisites
- Python (3.11.3)
- Visual Studio (2022) with C++ workload
- MSYS2

# Limitations compared to other approaches
- OSRT can directly optimize for an objective without a depth constraint
- ...

# Running experiments
Expects the following repositories to be checked out in the parent directory:
- `../optimal-sparse-regression-tree-public` (https://github.com/ruizhang1996/optimal-sparse-regression-tree-public)
- `../regression-tree-benchmark` (https://github.com/ruizhang1996/regression-tree-benchmark)
- `../streed2` (not published at this time)

1. Build OSRT as described below
2. Build streed2 using Visual Studio in release mode (executable expected at `../streed2/out/build/x64-Release/STREED`)
3. Get the datasets in the correct format by running `python prepare_data.py`
4. Update the path to MSYS2 bin folder in `experiments/experiments.py`, e.g. `"C:\\msys64\\ucrt64\\bin\\;"`
5. Run `python experiments.py` to run the experiments

# Building OSRT
0. Install MSYS2 with the following packages
```
$ pacman -Qe
base 2022.06-1
filesystem 2023.02.07-1
mingw-w64-ucrt-x86_64-autotools 2022.01.16-1
mingw-w64-ucrt-x86_64-boost 1.81.0-6
mingw-w64-ucrt-x86_64-gcc 12.2.0-10
mingw-w64-ucrt-x86_64-make 4.4-2
mingw-w64-ucrt-x86_64-opencl-icd 2023.04.17-1
mingw-w64-ucrt-x86_64-python 3.10.11-1
mingw-w64-ucrt-x86_64-tbb 2021.9.0-1
msys2-runtime 3.4.6-2
```
1. Apply the osrt patch for MSYS2 compatibility `git apply osrt.patch` on https://github.com/ruizhang1996/optimal-sparse-regression-tree-public (commit 4803498)
2. `./autobuild --install`
3. Test with `gosdt experiments/datasets/airfoil/airfoil.csv experiments/configurations/config.json >> output.json`

# STreeD cli
```
Main Parameters.
        -task.
                String. Task to optimize.
                Default value: accuracy
                Allowed values: accuracy, f1-score, regression, group-fairness
        -mode.
                String. Mode of optimization. Direct or hypertuning the number of nodes and depth.
                Default value: direct
                Allowed values: direct, hyper
        -file.
                String. Location to the (training) dataset.
                Default value:
        -test-file.
                String. Location to the test dataset.
                Default value:
Optional
        -time. Float.
                Maximum runtime given in seconds.
                Default value: 3600
                Range = [0, 2.14748e+09]
        -max-depth.
                Integer. Maximum allowed depth of the tree, where the depth is defined as the largest number of *decision/feature nodes* from the root to any leaf. Depth greater than four is usually time consuming.
                Default value: 3
                Range = [0, 20]
        -max-num-nodes.
                Integer. Maximum number of *decision/feature nodes* allowed. Note that a tree with k feature nodes has k+1 leaf nodes.
                Default value: 7
                Range = [0, 2147483647]
        -max-num-features.
                Integer. Maximum number of features that are considered from the dataset (in order of appearance).
                Default value: 2147483647
                Range = [1, 2147483647]
        -num-instances.
                Integer. Number of instances that are considered from the dataset (in order of appearance).
                Default value: 2147483647
                Range = [1, 2147483647]
        -verbose. Boolean.
                Determines if the solver should print logging information to the standard output.
                Default value: 1
        -all-trees. Boolean.
                Instructs the algorithm to compute trees using all allowed combinations of max-depth and max-num-nodes. Used to stress-test the algorithm.
                Default value: 0
        -train-test-split. Float.
                The percentage of instances for the test set
                Default value: 0
                Range = [0, 1]
        -stratify. Boolean.
                Stratify the train-test split
                Default value: 1
        -min-leaf-node-size.
                Integer. The minimum size of leaf nodes
                Default value: 1
                Range = [1, 2147483647]
Algorithmic Parameters.
        -similarity-lower-bound. Boolean.
                Activate similarity-based lower bounding. Disabling this option may be better for some benchmarks, but on most of our tested datasets keeping this on was beneficial.
                Default value: 1
        -use-upper-bound. Boolean.
                Use upper bounding. Disabling this option may be better for some benchmarks, specifically when the number of objectives is high.
                Default value: 1
        -use-lower-bound. Boolean.
                Use lower bounding. Disabling this option may be better for some benchmarks, specifically when the number of objectives is high.
                Default value: 1
        -feature-ordering.
                String. Feature ordering strategy used to determine the order in which features will be inspected in each node.
                Default value: in-order
                Allowed values: in-order, gini
        -random-seed.
                Integer. Random seed used only if the feature-ordering is set to random. A seed of -1 assings the seed based on the current time.
                Default value: 4
                Range = [-1, 2147483647]
        -use-branch-caching. Boolean.
                Use branch caching to store computed subtrees.
                Default value: 1
        -use-dataset-caching. Boolean.
                Use dataset caching to store computed subtrees. Dataset-caching is more powerful than branch-caching but may required more computational time.
                Default value: 0
        -duplicate-factor.
                Integer. Duplicates the instances the given amount of times. Used for stress-testing the algorithm, not a practical parameter.
                Default value: 1
                Range = [1, 2147483647]
Objective Parameters.
        -cost-file.
                String. Location of the file with information about the cost-sensitive classification.
                Default value:
Optional
        -ppg-file.
                String. Location of the file with cost information for prescriptive policy generation.
                Default value:
Optional
        -test-ppg-file.
                String. Location of the test file with cost information for prescriptive policy generation (required when giving a test-file).
                Default value:
Optional
        -ppg-teacher-method.
                String. Type of teacher model for prescriptive policy generation.
                Default value: DM
                Allowed values: DM, IPW, DR
Optional
        -ppg-capacity-limit. Float.
                The maximum expected demand (percentage) during training.
                Default value: 1
                Range = [0, 1]
        -fairness-type.
                String. Type of fairness for group fairness constraint.
                Default value: demographic-parity
                Allowed values: demographic-parity, equality-of-opportunity
Optional
        -discrimination-limit. Float.
                The maximum allowed percentage of discrimination in the training tree
                Default value: 1
                Range = [0, 1]
```