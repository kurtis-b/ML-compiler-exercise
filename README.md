# ML compiler exercise

## Goals and content

This is a *experimenting* and *simple* MLIR pipeline to lower ML models from PyTorch to x86. It is about to become an exercise 
for a new lecture at RWTH Aachen.

The repo provides some PyTorch models (small sample models and real world models) and provides the following:
1. Import models from PyTorch to MLIR using [torch-mlir](https://github.com/llvm/torch-mlir)
2. Use existing MLIR passes to lower from entry level IR's (linalg, arith, ...) to llvm ir
3. Create object file
4. Call the model via a function call in C++

Additionally, I created a pass that converts linalg.matmul's to [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) matrix-multiplication function calls. Further, I target an Nvidia GPU (sm_90) to launch specific kenrels on the GPU (This currently only works for sample and the mnist model). 

*Warning*: Some instructions are RWTH cluster specific, e.g. paths.

## Tutorial content

1. [Introduction](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter1.md)
2. [Getting started and project setup](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter2.md)
3. [Importing PyTorch models to torch-mlir](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter3.md)
4. [Lowering models to x86 machine code](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter4.md)
5. [Integration of OpenBLAS for Matrix Multiplications](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter5.md)
6. [Integration of real CNN (ResNet18) and transformer model (google-flan-t5)](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter6.md)
7. [Targeting Nvidia GPU](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter7.md)
8. *Appendix*: [An overview of IREE](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/iree_appendix.md)

## Build
See [Chapter 2](https://github.com/DavidGinten/ML-compiler-exercise/blob/main/docs/Chapter2.md) in the docs.

## References
- [Official MLIR website](https://mlir.llvm.org/) 
- [MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) by Jeremy Kun
- [MLIR tutorial with GPU compilation](https://www.stephendiehl.com/tags/mlir/) by Stephen Diehl
