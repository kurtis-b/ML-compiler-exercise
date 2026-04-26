# Chapter 6: Targeting An Nvidia GPU

This chapter documents the CUDA-on-WSL path. Start with the CPU setup in
[Chapter 2](Chapter2.md), then add the CUDA toolkit and rebuild torch-mlir with
NVPTX/CUDA support.

## CUDA-On-WSL Prerequisites

WSL should see the GPU through the Windows NVIDIA driver:

```bash
nvidia-smi
```

Install the CUDA toolkit inside WSL so `nvcc` is available:

```bash
nvcc --version
```

If `nvcc` is not on `PATH`, set either `CUDA_HOME` or `CUDACXX` before running
GPU builds:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX="$CUDA_HOME/bin/nvcc"
```

If you cannot use `sudo`, NVIDIA's runfile installer can install the toolkit to
a user-writable path without installing a driver:

```bash
bash cuda_<version>_linux.run \
  --silent \
  --toolkit \
  --toolkitpath="$HOME/cuda-toolkit" \
  --no-man-page \
  --override

export CUDA_HOME="$HOME/cuda-toolkit"
export CUDACXX="$CUDA_HOME/bin/nvcc"
```

The shared pipeline helper also checks `/usr/local/cuda/bin/nvcc` and the WSL
driver libraries under `/usr/lib/wsl/lib`.

## Build torch-mlir With CUDA Support

From `externals/torch-mlir`, configure a CUDA-capable build:

```bash
cd externals/torch-mlir
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}

cmake -GNinja -S externals/llvm-project/llvm -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython_FIND_VIRTUALENV=ONLY \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DMLIR_ENABLE_CUDA_RUNNER=ON \
  -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
  -DMLIR_ENABLE_NVPTXCOMPILER=ON \
  -DCUDACXX="$CUDACXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
  -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON

cmake --build build -- -j6
cd ../..
```

On this WSL machine, CUDA 12.3 provides `libnvptxcompiler_static.a` but not
`libnvfatbin_static.a`, so this torch-mlir revision cannot configure with
`MLIR_ENABLE_NVPTXCOMPILER=ON`. The ptxas fallback path configures with:

```bash
-DMLIR_ENABLE_NVPTXCOMPILER=OFF
```

That still requires `LLVM_TARGETS_TO_BUILD` to include `NVPTX` and
`MLIR_ENABLE_CUDA_RUNNER=ON`.

Confirm the expected cache flags are enabled:

```bash
cmake -LA -N externals/torch-mlir/build | grep -E 'MLIR_ENABLE_CUDA_RUNNER|MLIR_ENABLE_NVPTXCOMPILER|LLVM_TARGETS_TO_BUILD'
```

## CUDA Architecture

GPU scripts use `MLIR_CUDA_ARCH` as the numeric compute capability, for example
`86` for `sm_86` or `90` for `sm_90`.

The helper tries to auto-detect this with `nvidia-smi`. If detection fails, set
it explicitly:

```bash
export MLIR_CUDA_ARCH=86
```

## Run GPU Pipelines

GPU validation is opt-in:

```bash
source .venv/bin/activate
export TORCH_MLIR_SOURCE_DIR="$PWD/externals/torch-mlir"
export TORCH_MLIR_BUILD_DIR="$PWD/externals/torch-mlir/build"
bash src/validate_wsl.sh --gpu
```

The intended CUDA-on-WSL GPU coverage is:

- `sample`
- `mnist`
- `cnn`
- `resnet18`
- `flan-t5-small`

If a model fails because of lowering semantics rather than path/tool discovery,
keep the script failure explicit and document the model-specific blocker.
