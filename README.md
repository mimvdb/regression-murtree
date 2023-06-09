# Regression MurTree
This repository contains scripts to run experiments comparing the runtime of regression tree training algorithms

## Notes
- OSRT can directly optimize for an objective without a depth constraint
- OSRT will crash if regularization factor is set to 0

# Linux
## Prerequisites
- CMake
- Python (see requirements.txt)

## Running experiments
Expects the following repositories to be checked out in the parent directory:
- `../optimal-sparse-regression-tree-public` (https://github.com/ruizhang1996/optimal-sparse-regression-tree-public)
- `../regression-tree-benchmark` (https://github.com/ruizhang1996/regression-tree-benchmark)
- `../streed2` (not published at this time)

1. Build OSRT with `./autobuild --install` after installing dependencies `sudo apt-get install libboost-dev libboost-all-dev libgmp-dev libgmp10 ocl-icd-opencl-dev libtbb-dev` and `automake --add-missing`
2. Build streed2 by following the linux instructions in the README
3. Get the datasets in the correct format by running `python prepare_data.py`
4. Run `python experiments.py` to run the experiments
5. Run `python plot_experiments.py` to plot the results, see the results in `experiments/figures/**/*.[png,svg]`

# Windows (may be outdated)
## Prerequisites
- Python (3.11.3)
  - seaborn (matplotlib, numpy)
  - scikit-learn
- Visual Studio (2022) with C++ workload
- MSYS2

## Running experiments
Expects the following repositories to be checked out in the parent directory:
- `../optimal-sparse-regression-tree-public` (https://github.com/ruizhang1996/optimal-sparse-regression-tree-public)
- `../regression-tree-benchmark` (https://github.com/ruizhang1996/regression-tree-benchmark)
- `../streed-regression` (not published at this time)

1. Build OSRT as described below
2. Build streed2 using Visual Studio in release mode (check that configuration is Release, not RelWithDebInfo) (executable expected at `../streed2/out/build/x64-Release/STREED`)
3. Get the datasets in the correct format by running `python prepare_data.py`
4. Update the path to MSYS2 bin folder in `experiments/experiments.py`, e.g. `"C:\\msys64\\ucrt64\\bin\\;"`
5. Run `python experiments.py` to run the experiments
6. Run `python plot_experiments.py` to plot the results, see the results in `experiments/figures/**/*.png`

## Building OSRT
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


# Delftblue

# Load required modules to build OSRT/STreeD and run experiments
module load 2022r2
module load python
module load gcc/11.2.0
module load cmake
module load boost
module load gmp
module load intel/oneapi-all