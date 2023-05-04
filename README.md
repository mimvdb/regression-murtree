# Regression MurTree

# Comparison with OSRT
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
1. Apply the osrt patch for MSYS2 compatibility `git apply osrt.patch` on https://github.com/ruizhang1996/optimal-sparse-regression-tree-public
2. ./autobuild --install
3. cat experiments/datasets/airfoil/airfoil.csv | gosdt experiments/configurations/config.json >> output.json