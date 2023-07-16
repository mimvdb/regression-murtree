# Regression MurTree
This repository contains the experiments performed for the bachelor thesis "Optimal Regression Trees via Dynamic Programming: Optimization techniques for learning Regression Trees" (http://resolver.tudelft.nl/uuid:377edc0f-00b9-4481-840f-0fde43c494b9), as part of the Research Project 2023/Q4 (https://cse3000-research-project.github.io/2023/Q4) at the TU Delft.

For the state of the repository as it was at the end of the research project, see the tag `research-project`.

# Linux
## Prerequisites
- CMake
- Python (see requirements.txt)

## Preparing data
```
cd data
./fetch.py
./clean.py
./binarize.py
```

## Running experiments
Expects the following repositories to be checked out in the parent directory:
- `../optimal-sparse-regression-tree-public` (https://github.com/ruizhang1996/optimal-sparse-regression-tree-public)
- `../regression-tree-benchmark` (https://github.com/ruizhang1996/regression-tree-benchmark)
- `../streed2` (not published at this time)

1. Build OSRT with `./autobuild --install` after installing dependencies `sudo apt-get install libboost-dev libboost-all-dev libgmp-dev libgmp10 ocl-icd-opencl-dev libtbb-dev` and `automake --add-missing`
2. Build streed2 by following the linux instructions in the README
3. Prepare datasets ...
4. Run `python setup_scalability.py` to intialize the experiment
5. Run `python lab_runner.py` to run the experiments with multiple processes or on DelftBlue, or run `python sync_runner.py` to run single-threaded.
6. ~~Run `python plot_experiments.py` to plot the results, see the results in `experiments/figures/**/*.[png,svg]`~~

# Windows (outdated)
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

https://doc.dhpc.tudelft.nl/delftblue/

## Load required modules to build OSRT/STreeD and run experiments

For all:
```bash
module load 2022r2
```
For osrt:
```bash
module load boost
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
