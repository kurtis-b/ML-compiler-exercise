# Chapter 2: Getting started and project setup
*Warning*: Instructions are partially RWTH cluster-specific.

## Project Structure
The project is organized as follows:
- **externals/** - Contains torch-mlir and, along with it, an in-tree checkout of LLVM/MLIR.
- **lib/** - Stores our custom passes.
- **src/** - Responsible for importing PyTorch models into MLIR, running the pass pipeline, and executing/benchmarking the models using C/C++.
- **tools/** - Contains the pass pipeline configuration.

Aside:
In a typical setup, you would also have an include/ directory holding header files, while lib/ contains the implementation files. The lib/ directory often includes subdirectories for different categories of passes (e.g., lib/Transform). In this project, we place both headers and implementation files together in lib/, since we currently have only a small number of passes (just one at the moment).

## Build
Initalize the submodules (torch-mlir, llvm-project)
`git submodule update --init --recursive`

Set up a python virtual environment
```
python3 -m venv dev
source dev/bin/activate
pip install --upgrade pip
```

Install the requirements in the root:
`python -m pip install -r requirements.txt`

Go to [torch-mlir](../externals/torch-mlir) to follow the documentation in [development guide](../externals/torch-mlir/docs/development.md) for next steps. Instructions are given below as well, but could be slightly different with updates to the torch-mlir repository.

Install the requirements.
`python -m pip install -r requirements.txt -r torchvision-requirements.txt`

Configuration for building torch-mlir with llvm in-tree.
```
cmake -GNinja -Bbuild  \
  `# Enables "--debug" and "--debug-only" flags for the "torch-mlir-opt" tool` \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  `# For building LLVM "in-tree"` \
  externals/llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  `# use clang` \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  `# use ccache to cache build results` \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON
```

Then build with: `cmake --build build`. Append `-j8` to set the number of parallel threads for building.

In dev/bin/activate, add the following (adapt the path if necessary):
```
# Add torch-mlir-opt to PATH
export PATH="~/ml-compiler-exercise/externals/torch-mlir/build/bin/:$PATH"

# Add MLIR Python bindings and Set up Python Environment to export the built Python packages
export PYTHONPATH=~/ml-compiler-exercise/externals/torch-mlir/build/tools/mlir/python_packages/mlir_core:~/ml-compiler-exercise/externals/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:~/ml-compiler-exercise/externals/torch-mlir/test/python/fx_importer
```

Test the build works by going to [projects/pt1](../externals/torch-mlir/projects/pt1/) and running
`./tools/e2e_test.sh`