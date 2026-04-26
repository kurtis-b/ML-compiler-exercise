# Chapter 8: Triton MLIR Internals Walkthrough

Triton has real MLIR dialects, including Triton IR (`tt`) and TritonGPU
(`ttg`). In this repo, the goal is to inspect Triton's generated IR first, not
to hand-author Triton dialect programs.

Primary references:

- [Triton MLIR dialects](https://triton-lang.org/main/dialects/dialects.html)
- [Triton `tt` dialect](https://triton-lang.org/main/dialects/TritonDialect.html)
- [TritonGPU ops](https://triton-lang.org/main/dialects/TritonGPUOps.html)
- [Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)

## What This Is And Is Not

This chapter is about observing Triton's compiler pipeline. It is not claiming
that Triton dialects are upstream LLVM MLIR dialects, and it is not saying you
should start by writing `.ttir` by hand.

The practical workflow is:

1. Write a small Triton kernel in Python.
2. Validate it against PyTorch.
3. Enable IR dumping.
4. Inspect the files produced by your installed Triton version.
5. Map those files back to the Python kernel concepts.

## Dump IR For The Vector-Add Example

Run:

```bash
python src/gpu-tutorial/triton-vector-add/vector_add.py --size 1024 --dump-ir
```

The script sets repo-local locations before importing Triton:

```bash
TRITON_KERNEL_DUMP=1
TRITON_ALWAYS_COMPILE=1
TRITON_DUMP_DIR=.triton-dumps/vector-add
TRITON_CACHE_DIR=.triton-cache
TRITON_HOME=.triton-home
```

Triton versions differ in exact filenames and emitted stages, so the script does
not promise fixed artifact names. Instead, it lists the files it actually sees.
Depending on backend and version, you may see files with names or suffixes such
as TTIR, TritonGPU IR, LLVM IR, PTX, CUBIN, or JSON metadata.

For a noisier compiler-developer view, Triton's own repository documents:

```bash
MLIR_ENABLE_DUMP=1
MLIR_DUMP_PATH=<directory>
```

That emits IR around MLIR passes and can be much more verbose than
`TRITON_KERNEL_DUMP`.

## What To Look For

In the vector-add dump, look for operations corresponding to:

- program ID calculation;
- vector offsets;
- masked loads;
- elementwise add;
- masked store.

In the softmax dump, look for:

- row program mapping;
- masked load with `-inf` for inactive lanes;
- max and sum reductions;
- exponential;
- final normalized store.

The names and lowering details can change across Triton releases. Treat the dump
as compiler evidence for your installed version, not as a stable public IR
contract.

## Relationship To MLIR GPU

MLIR GPU and Triton both compile GPU work, but they enter at different levels:

- MLIR GPU examples expose `gpu.launch`, `gpu.module`, host/runtime lowering,
  and backend serialization directly.
- Triton exposes a Python kernel DSL and lets Triton's compiler decide the
  lower-level Triton/TritonGPU/LLVM/PTX path.

Learning both is useful: MLIR GPU teaches the compiler substrate, while Triton
teaches productive custom-kernel authoring.
