# ML compiler exercise

This tutorial demonstrates an end-to-end MLIR flow for lowering PyTorch and
Hugging Face models to executable CPU and NVIDIA GPU binaries. It uses upstream
[torch-mlir](https://github.com/llvm/torch-mlir), MLIR lowering pipelines, a
small custom OpenBLAS rewrite pass, and model-specific C++/pybind runners.

## Supported WSL Flows

The WSL CPU flow is supported for:

- `sample`
- `mnist`
- `cnn`
- `resnet18`
- `bert-base-uncased`
- `flan-t5-small`
- `gpt`

The CUDA-on-WSL flow is intended for:

- `sample`
- `mnist`
- `cnn`
- `resnet18`
- `flan-t5-small`

GPU validation requires a CUDA-enabled torch-mlir build and `nvcc` inside WSL.

The `gpt` pipeline defaults to `sshleifer/tiny-gpt2` so the end-to-end lowering
is practical on a local machine. Use `GPT_MODEL_NAME=gpt2` to try full GPT-2;
that path is much heavier and can produce very large intermediate MLIR.

## Build

See [Chapter 2](docs/Chapter2.md) for the CPU setup and top-level build. See
[Chapter 6](docs/Chapter6.md) for CUDA-on-WSL setup.

Quick WSL validation after torch-mlir is built:

```bash
export TORCH_MLIR_BUILD_DIR="$PWD/externals/torch-mlir/build"
export TORCH_MLIR_SOURCE_DIR="$PWD/externals/torch-mlir"
bash build.sh
cmake --build build-ninja --target check-mlir-tutorial -- -j6
bash src/validate_wsl.sh
```

Run GPU validation explicitly:

```bash
bash src/validate_wsl.sh --gpu
```

## Tutorial Content

1. [Introduction](docs/Chapter1.md)
2. [Getting started and project setup](docs/Chapter2.md)
3. [Importing PyTorch models to torch-mlir](docs/Chapter3.md)
4. [Lowering models to x86 machine code](docs/Chapter4.md)
5. [Integration of OpenBLAS for Matrix Multiplications](docs/Chapter5.md)
6. [Targeting an Nvidia GPU](docs/Chapter6.md)

Appendix: [An overview of IREE](docs/iree_appendix.md)

## References

- [Official MLIR website](https://mlir.llvm.org/)
- [MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) by Jeremy Kun
- [MLIR tutorial with GPU compilation](https://www.stephendiehl.com/tags/mlir/) by Stephen Diehl
