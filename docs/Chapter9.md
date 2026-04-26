# Chapter 9: Heterogeneous Compute Walkthrough

Heterogeneous compute means one program uses more than one kind of processor,
usually a CPU host plus one or more accelerators. In this repo, that mostly
means CPU host code launching NVIDIA GPU kernels.

Primary references:

- [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR NVGPU dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [MLIR Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
- [IREE](https://iree.dev/)

## What This Is And Is Not

This chapter is the map. It is not another kernel language and it is not saying
IREE is required for this repo.

Use it to understand how the pieces fit:

- CPU host code owns process startup, memory setup, and kernel launches.
- GPU kernels run many lightweight threads over device-visible memory.
- MLIR GPU/NVGPU/NVVM represent progressively lower-level GPU compiler IR.
- Triton is a productive DSL for writing custom GPU kernels.
- torch-mlir imports PyTorch/model programs into MLIR.
- IREE is a larger MLIR-based compiler/runtime project for deployment-oriented
  heterogeneous execution.

## The Host/Device Split

A GPU program normally has at least two sides:

- Host side: allocate buffers, copy or map memory, choose launch dimensions,
  launch kernels, synchronize, and check results.
- Device side: execute a kernel across blocks, threads, lanes, and memory
  spaces.

The MLIR vector-add tutorial makes this explicit with `gpu.alloc`,
`gpu.memcpy`, `gpu.launch`, and `gpu.dealloc`. The Triton examples hide more of
that machinery, but the same conceptual split still exists.

## Where Each Tool Fits

MLIR GPU is a compiler IR path. It is useful when you want to study or build
lowering pipelines, scheduling, outlining, target lowering, and runtime linking.

Triton is a kernel authoring path. It is useful when you want to write a custom
operation such as softmax, matmul, layer norm, or a fused model fragment without
writing CUDA C++ directly.

torch-mlir is a model import/lowering path. It is useful when you want to bring
PyTorch programs into MLIR and then experiment with compiler transformations.

IREE is a system path. It combines compiler and runtime ideas for dispatching
work to different hardware targets. Use it as a reference when you want to learn
how MLIR-based heterogeneous compilation becomes an application deployment
stack.

## Path Forward For Your Own Work

1. Modify constants and shapes in vector add, then inspect how MLIR and Triton
   artifacts change.
2. Write simple elementwise kernels: add, multiply, ReLU, clamp, and bias add.
3. Write bandwidth-bound kernels: row sum, softmax, and layer norm forward.
4. Write tiled compute kernels: small matmul, then fused matmul plus activation.
5. Compare one operation implemented through MLIR GPU and Triton.
6. Connect a custom kernel to a model piece, replacing one operation with your
   own implementation or lowering.
7. Start compiler work: add a small MLIR rewrite/pass, then use transform-style
   scheduling concepts for tiling and mapping.
8. Explore IREE concepts: dispatches, executable targets, host/device runtime
   integration, and deployment-oriented validation.

Keep the first version tiny. Correctness first, then measurement, then
optimization.

## Validation Habits

For every heterogeneous experiment:

- keep a CPU or PyTorch reference;
- print a max absolute or relative error;
- preserve generated IR or binary artifacts when studying compiler behavior;
- separate smoke tests from benchmarks;
- label anything version-sensitive or unvalidated.

The goal is not just to make a kernel run. The goal is to know what ran, where it
ran, how it was lowered, and how you know it was correct.
