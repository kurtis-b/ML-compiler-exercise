# MLIR GPU vector add

This is a tiny MLIR GPU walkthrough example. It is not produced by torch-mlir;
it is hand-written so you can inspect the host/device boundary directly.

```bash
bash run.sh
```

The scripts write staged artifacts to `build/`:

- `01-verified.mlir`: parsed and verified input.
- `02-outlined.mlir`: after `gpu-kernel-outlining`.
- `03-nvvm-binary.mlir`: after NVVM lowering and GPU binary serialization.
- `04-verified-nvvm-binary.mlir`: final MLIR verification copy.
- `vector_add.ll`, `vector_add.o`, and `vector_add_runner`: LLVM IR, object,
  and native runner.

The runner exits non-zero if the GPU output does not match the CPU reference.
