# Chapter 6: MLIR GPU Walkthrough

This chapter is the low-level GPU path. It shows how MLIR represents host code,
device code, kernel launches, and binary embedding before anything becomes a
native executable.

Primary references:

- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR NVGPU dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)

## What This Is And Is Not

This is a walkthrough of MLIR's `gpu` and NVVM lowering path on CUDA-on-WSL.
It is not Triton, and it is not a full model compiler by itself.

The tiny example in `src/gpu-tutorial/mlir-vector-add/` is hand-written MLIR so
you can see the concepts directly:

- Host function: `func.func @vector_add`.
- Device allocation/copies: `gpu.alloc`, `gpu.memcpy`, `gpu.dealloc`.
- Kernel launch: `gpu.launch`.
- Kernel outlining: `gpu-kernel-outlining` creates a `gpu.module` and GPU
  kernel function.
- CUDA lowering: `gpu-lower-to-nvvm-pipeline` lowers GPU code toward NVVM.
- Binary embedding: `gpu-module-to-binary` serializes the GPU module.
- Host linking: the C++ runner links MLIR runner utilities and CUDA runtime
  libraries.

The existing model GPU directories, such as `src/sample/gpu/`, are advanced
examples that start from tensor/linalg model IR and then route through a similar
GPU backend path. The tiny vector-add path is easier to study first.

## CUDA-On-WSL Prerequisites

WSL should see the GPU through the Windows NVIDIA driver:

```bash
nvidia-smi
```

Install the CUDA toolkit inside WSL so `nvcc` is available:

```bash
nvcc --version
```

If `nvcc` is not on `PATH`, set either `CUDA_HOME` or `CUDACXX`:

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

The MLIR GPU tutorial needs MLIR tools and MLIR's CUDA runtime support. From
`externals/torch-mlir`, configure a CUDA-capable build:

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

cmake --build build -- -j2
cd ../..
```

On some CUDA toolkit versions, `MLIR_ENABLE_NVPTXCOMPILER=ON` may fail because
the toolkit does not ship every static library expected by this MLIR revision.
The ptxas fallback path is:

```bash
-DMLIR_ENABLE_NVPTXCOMPILER=OFF
```

That fallback still needs `LLVM_TARGETS_TO_BUILD` to include `NVPTX` and
`MLIR_ENABLE_CUDA_RUNNER=ON`.

Confirm the relevant cache flags:

```bash
cmake -LA -N externals/torch-mlir/build | grep -E 'MLIR_ENABLE_CUDA_RUNNER|MLIR_ENABLE_NVPTXCOMPILER|LLVM_TARGETS_TO_BUILD'
```

## Run The Tiny MLIR GPU Tutorial

From the repo root:

```bash
GPU_TUTORIAL_SIZE=1024 bash src/validate_gpu_tutorial.sh
```

Or run the MLIR lane directly:

```bash
cd src/gpu-tutorial/mlir-vector-add
bash run.sh
```

The lane is intentionally staged:

```text
vector_add_gpu.mlir
  -> 01-verified.mlir
  -> 02-outlined.mlir
  -> 03-nvvm-binary.mlir
  -> 04-verified-nvvm-binary.mlir
  -> vector_add.ll
  -> vector_add.o
  -> vector_add_runner
```

Each MLIR stage is verified with `mlir-opt`. The C++ runner compares the GPU
result against a CPU reference and fails if the max absolute error is greater
than `1e-6`.

## How To Read The Stages

Start with `vector_add_gpu.mlir`. Notice that `gpu.launch` is embedded in a
normal host `func.func`, and the launch body uses block and thread identifiers
from the launch operation.

Then inspect `build/02-outlined.mlir`. `gpu-kernel-outlining` should have moved
the launch body into a `gpu.module`/kernel function and replaced the original
launch with a `gpu.launch_func` style host-side launch.

Then inspect `build/03-nvvm-binary.mlir` or `build/04-verified-nvvm-binary.mlir`.
At this point the device side has been lowered toward NVVM and serialized into a
GPU binary operation. The host side still needs LLVM translation and native
linking.

The important correction to keep in mind: MLIR does not magically discover GPU
parallelism here. The tiny example already contains explicit GPU parallelism in
`gpu.launch`. The model GPU pipelines first have to create/tile parallel loops
before using a similar GPU backend.

## Relationship To The Existing Model GPU Scripts

`src/sample/gpu/run_mlir_pipeline.sh` starts from linalg-style model IR, applies
loop/tile transformations, converts affine loops to GPU, outlines kernels, and
then follows the same broad backend idea: GPU dialect to NVVM, GPU binary
serialization, LLVM translation, object generation, and C++ host linking.

Use the tiny vector add first. Use the model GPU scripts after you are
comfortable reading the artifacts.

## Heavier Validation

The default validation does not run full model GPU coverage. When you are not
using the PC interactively, the heavier opt-in command is:

```bash
bash src/validate_wsl.sh --gpu
```

Do not treat a model GPU pipeline as supported until that exact pipeline has
been run and compared against its reference output.
