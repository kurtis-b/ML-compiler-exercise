# Chapter 2: Getting Started And Project Setup

This chapter documents the WSL CPU setup. CUDA-on-WSL setup is covered in
[Chapter 6](Chapter6.md).

## Project Structure

- `externals/` contains upstream torch-mlir and its LLVM/MLIR checkout.
- `lib/` contains the tutorial pass implementation.
- `src/` contains model import, lowering, compile, run, benchmark, and validation scripts.
- `tools/` contains the `tutorial-opt` driver.
- `tests/` contains the lit regression test for the tutorial pass.

## WSL CPU Prerequisites

Install the usual native build tools, Python, Ninja, CMake, and OpenBLAS:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libopenblas-dev \
  ninja-build \
  python3 \
  python3-venv
```

Clone the repo and initialize submodules:

```bash
git clone <repo-url> ML-compiler-exercise
cd ML-compiler-exercise
git submodule sync --recursive
git submodule update --init --recursive
```

The `externals/torch-mlir` submodule should point at upstream
`https://github.com/llvm/torch-mlir.git`.

## Python Environment

Create and activate a repo-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r externals/torch-mlir/requirements.txt \
  -r externals/torch-mlir/torchvision-requirements.txt
python -m pip install pybind11 transformers sentencepiece
```

Heavy model validation may download Hugging Face assets. Set `HF_TOKEN` if you
want authenticated downloads and higher rate limits.

## Build torch-mlir For CPU

Configure torch-mlir with LLVM/MLIR in-tree:

```bash
cd externals/torch-mlir
cmake -GNinja -S externals/llvm-project/llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON

cmake --build build -- -j6
cd ../..
```

Use fewer or more build jobs by changing `-j6`.

## Build This Tutorial

`build.sh` is the supported top-level entrypoint. It is incremental and does not
delete `build-ninja` on each run.

```bash
source .venv/bin/activate
export TORCH_MLIR_SOURCE_DIR="$PWD/externals/torch-mlir"
export TORCH_MLIR_BUILD_DIR="$PWD/externals/torch-mlir/build"
export BUILD_JOBS=6

bash build.sh
cmake --build build-ninja --target check-mlir-tutorial -- -j6
```

## Validate CPU Pipelines

Run the opt-in WSL validation script:

```bash
bash src/validate_wsl.sh
```

It validates:

- `sample`
- `mnist`
- `cnn`
- `resnet18`
- `bert-base-uncased`
- `flan-t5-small`
- `gpt`

Numeric models compare MLIR output against PyTorch output. Flan and GPT compare
generated token IDs. The GPT flow defaults to `sshleifer/tiny-gpt2`; use
`GPT_MODEL_NAME=gpt2` to try full GPT-2.

## Environment Knobs

- `TORCH_MLIR_BUILD_DIR` defaults to `externals/torch-mlir/build`.
- `TORCH_MLIR_SOURCE_DIR` defaults to `externals/torch-mlir`.
- `OPENBLAS_DIR` defaults to `openblas`.
- `OPENBLAS_LIB_DIR` overrides OpenBLAS library discovery directly.
- `BUILD_JOBS` controls the build parallelism used by `build.sh`.
- `GPT_MODEL_NAME`, `GPT_PROMPT`, and `GPT_MAX_NEW_TOKENS` customize GPT validation.
