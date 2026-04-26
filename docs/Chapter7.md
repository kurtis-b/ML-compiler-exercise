# Chapter 7: Triton Kernel Walkthrough

Triton is a Python-embedded kernel language and compiler for writing GPU
kernels at a higher level than CUDA C++. It is a different path from MLIR's
hand-written `gpu` dialect examples, even though Triton's compiler internally
uses MLIR dialects.

Primary references:

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)

## What This Is And Is Not

This chapter teaches Triton kernel authoring: program IDs, masks, block sizes,
loads, stores, reductions, and `tl.dot`.

It is not a replacement for the MLIR GPU chapter. Triton hides much of the
lower-level host/device lowering machinery so you can write kernels directly in
Python.

## Setup

Triton is treated as a Python dependency, not a vendored submodule:

```bash
python -m pip install triton
```

You also need a CUDA-visible PyTorch install for the examples:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

The validator skips the Triton lane if PyTorch, Triton, or `torch.cuda` is not
available.

## Lesson 1: Vector Add

Run:

```bash
python src/gpu-tutorial/triton-vector-add/vector_add.py --size 1024
```

Read the kernel in `src/gpu-tutorial/triton-vector-add/vector_add.py`:

- `tl.program_id(axis=0)` selects which block of elements this program handles.
- `tl.arange(0, BLOCK_SIZE)` creates vector offsets inside the program.
- `mask = offsets < n_elements` prevents out-of-bounds memory access.
- `tl.load` and `tl.store` perform vectorized memory operations under the mask.
- The script compares against `x + y` with `torch.allclose`.

This example is bandwidth-shaped: the work is simple, and most of the lesson is
about mapping contiguous elements to programs and masks.

## Lesson 2: Row-Wise Softmax

Run:

```bash
python src/gpu-tutorial/triton-softmax/softmax.py --rows 4 --cols 32
```

This kernel assigns one row to each Triton program. The kernel demonstrates:

- masked loads for rows that do not fill the whole power-of-two block;
- subtracting the row maximum for numerical stability;
- `tl.max`, `tl.exp`, and `tl.sum` reductions inside one program;
- writing the fused softmax result back once.

The script compares against `torch.softmax(x, dim=1)` and prints the maximum
absolute error.

## Lesson 3: Small Matmul Progression

After vector add and softmax, implement a tiny matmul as a local exercise:

1. Start with one program per output tile.
2. Use two-dimensional program IDs for tile row and tile column.
3. Build row and column offset vectors.
4. Use masked `tl.load` for the A and B tiles.
5. Accumulate with `tl.dot`.
6. Store the output tile with a mask.

Keep the first shape tiny, for example `M=N=K=32`, and compare against
`torch.matmul`. Only add autotuning after the non-autotuned kernel is correct.

## Correctness Checks

Every Triton tutorial script should:

- use fixed seeds;
- compare against a PyTorch reference;
- print the maximum absolute error;
- fail non-zero on mismatch;
- avoid benchmarking unless explicitly requested.

This is important because a fast wrong kernel is just a small space heater with
opinions.
