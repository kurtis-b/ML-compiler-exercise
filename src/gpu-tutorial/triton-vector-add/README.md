# Triton vector add

This is the smallest Triton lane in the tutorial. It teaches one-dimensional
program IDs, masks, block sizes, and numerical comparison against PyTorch.

```bash
python vector_add.py --size 1024
python vector_add.py --size 1024 --dump-ir
```

`--dump-ir` sets repo-local dump/cache directories before importing Triton and
prints the files produced by the installed Triton version.
