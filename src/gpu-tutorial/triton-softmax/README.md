# Triton softmax

This example keeps one row per Triton program and one block covering the row.
It is intentionally tiny so you can study masking, reductions, and fusion
without benchmarking noise.

```bash
python softmax.py --rows 4 --cols 32
```

The script compares against `torch.softmax` and prints the maximum absolute
error.
