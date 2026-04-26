# GPU tutorial examples

These examples are intentionally small. They are meant to teach the shape of
the GPU compilation paths without running the heavier model pipelines.

## Examples

- `mlir-vector-add/` starts from hand-written MLIR with `gpu.launch`, lowers it
  through the MLIR GPU/NVVM path, links a tiny C++ host runner, and compares
  the GPU result with a CPU reference.
- `triton-vector-add/` starts from a small `triton.jit` kernel and compares the
  result with PyTorch.
- `triton-softmax/` shows a single-row-per-program softmax kernel with masking,
  reductions, and a PyTorch comparison.

Run the low-impact smoke test from the repo root:

```bash
GPU_TUTORIAL_SIZE=1024 bash src/validate_gpu_tutorial.sh
```

The script skips optional lanes when CUDA, CUDA-enabled torch-mlir, PyTorch
CUDA, or Triton are not available.
