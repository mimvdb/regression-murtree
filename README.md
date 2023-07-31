# Regression MurTree
This repository contains the experiments performed for the bachelor thesis "Optimal Regression Trees via Dynamic Programming: Optimization techniques for learning Regression Trees" (http://resolver.tudelft.nl/uuid:377edc0f-00b9-4481-840f-0fde43c494b9), as part of the Research Project 2023/Q4 (https://cse3000-research-project.github.io/2023/Q4) at the TU Delft.

For the state of the repository as it was at the end of the research project, see the tag `research-project`.

# Linux
## Prerequisites
- CMake
- Python (`pip install -r requirements.txt`)
- InterpretableAI (https://docs.interpretable.ai/stable/IAI-Python/installation/#Python-Auto-Install after installing python package)
- R (for GUIDE) (`sudo apt-get install r-base`)

## Preparing data
```sh
cd data
./fetch.py # Fetch datasets
./clean.py # Clean datasets to not contain missing values and only have numeric features
./binarize.py # Create binary features from continuous features
./prepare.py # Create train/test splits in all required formats
```

## Running experiments
Expects the following repositories to be checked out in the parent directory:
- `../optimal-sparse-regression-tree-public` (https://github.com/ruizhang1996/optimal-sparse-regression-tree-public)
- `../streed2` (not published at this time)

1. Build OSRT after installing dependencies `sudo apt-get install libgmp-dev libtbb-dev` with
```sh
mkdir build
cd build
cmake ..
cmake --build .
```
2. Build streed2 by following the linux instructions in the README
3. Prepare datasets (See above)
4. Run `python setup_scalability.py` to intialize the experiment
5. Run `python lab_runner.py` to run the experiments with multiple processes or on DelftBlue, or run `python sync_runner.py` to run single-threaded.
6. (If lab_runner was used) Run `python lab_aggregate.py` to aggregate the results from all processes in to a single csv.

## Plotting experiments
After preparing the data/running the experiments, run any of the plot scripts `python plot_*.py`.

# Delftblue

https://doc.dhpc.tudelft.nl/delftblue/

## Load required modules to build OSRT/STreeD and run experiments

For all:
```bash
module load 2022r2
```
For osrt:
```bash
module load gmp
module load intel/oneapi-all
```
For streed:
```bash
module load cmake
```
For experiments:
```bash
module load python
module load py-numpy
module load py-scikit-learn
module load py-pillow
module load py-pip
```
Install python packages as in requirements.txt, make sure pandas is compatible with numpy (use 1.3.4), copy experiments folder to machine, change email/account in runner.py and run `./runner.py --all`.

## Fix library and include path for OSRT
Remove the stdin check from main.cpp.
Add the include path of libgmp module (hint: find with echo $LD_LIBRARY_PATH) to `Makefile.am`

```sh
export LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
automake --add-missing
./autobuild --configure --force
./configure
make gosdt` (, not `./autobuild --build)
```
