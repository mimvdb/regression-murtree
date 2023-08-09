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

## Setup
- Add `gurobi.lic` file to run MIP methods. See the delftblue docs recipe.
- Create a venv environment for installing packages (don't install requirements.txt yet)
- Add the required modules to `.bashrc` so they get loaded for experiments, the following is (more than) enough to run the experiments.
```
module load 2022r2
module load python
module load py-numpy
module load py-scikit-learn
module load py-pip
module load boost
module load cmake
module load gmp
module load intel/oneapi-all

. "/home/{username}/.venv/bin/activate"
```
- Relog, or manually load the modules/venv and build the required methods by following their instructions. For building, use `module load gcc` to build with a more recent gcc version.
- Install packages not yet loaded by modules in `requirements-delftblue.txt`

## Workflow
Fork this repo on the server in `/scratch/{username}`, but create datasets locally and copy the (zipped) result to the server. Adjust the paths to executables in `sync_runner.py`, and change to DelftBlueEnvironment and set the slurm account/notification email/resource limits in `lab_runner.py`.

Then for each experiment:
- Run `setup_X.py` locally. Copy experiments.json to server
- Choose the batching factor by changing the amount of runs in a chunk in `lab_runner.py`
- Remove experiment directories between runs to avoid confusion:
```sh
rm -r experiment/
rm -r experiment-grid-steps/
rm -r tmp/
```
- Run experiment `./lab_runner.py --all`
- Check progress with `squeue --user={username}`, if needed, cancel with `scancel {taskid}`
- Run `./lab_finderr.py` to aggregate all error logs from experiments into `log.err`. Check `experiment-grid-steps/slurm.err` for slurm errors (like cancelled jobs due to timeout)
- Run `./lab_aggregate.py` to aggregate the results to `results.csv`
- Copy the results to your local system and run `./setup_failed.py` and check the output/`new_experiments.json` for any experiments that did not have a matching result. If any, rerun and combine with `./lab_aggregate.py --in-dir {dir_with_both_results.csv}` and repeat.

## OSRT
When building OSRT, remove the stdin check from main.cpp to prevent it getting stuck when run from a python script. (So it always uses the provided file for the config). Add the include path of libgmp module (Find the path with `echo $LD_LIBRARY_PATH`) to `Makefile.am`. Then, to build, run:
```sh
export LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
automake --add-missing
./autobuild --configure --force
./configure
make gosdt` (, not `./autobuild --build)
```
Building with cmake might be slightly different.