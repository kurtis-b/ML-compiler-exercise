#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Triton row-wise softmax tutorial")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--dump-ir", action="store_true", help="dump Triton compiler IR to repo-local folders")
    return parser.parse_args()


def configure_dumping(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    dump_dir = repo_root / ".triton-dumps" / "softmax"
    cache_dir = repo_root / ".triton-cache"
    home_dir = repo_root / ".triton-home"
    if args.dump_ir:
        dump_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        home_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
        os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
        os.environ.setdefault("TRITON_DUMP_DIR", str(dump_dir))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir))
        os.environ.setdefault("TRITON_HOME", str(home_dir))
    return dump_dir, cache_dir


args = parse_args()
dump_dir, cache_dir = configure_dumping(args)

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit(f"PyTorch and Triton are required for this example: {exc}") from exc


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, stride: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row * stride

    values = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row_start + offsets, numerator / denominator, mask=mask)


def list_dumped_files(path: Path) -> None:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        print(f"No Triton dump files were observed under {path}")
        return
    print(f"Triton dump files under {path}:")
    for file in files:
        print(f"  {file.relative_to(path)}")


def main() -> None:
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows and --cols must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false; this example needs a CUDA-visible PyTorch install")

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    block_size = next_power_of_two(args.cols)
    softmax_kernel[(args.rows,)](x, out, args.cols, args.cols, BLOCK_SIZE=block_size)

    expected = torch.softmax(x, dim=1)
    max_abs_error = torch.max(torch.abs(out - expected)).item()
    if not torch.allclose(out, expected, atol=1.0e-5, rtol=1.0e-5):
        raise SystemExit(f"softmax mismatch: max_abs_error={max_abs_error:.8g}")

    print(
        f"Triton softmax rows={args.rows} cols={args.cols} "
        f"block_size={block_size} max_abs_error={max_abs_error:.8g}"
    )
    if args.dump_ir:
        list_dumped_files(dump_dir)
        print(f"Triton cache directory: {cache_dir}")


if __name__ == "__main__":
    main()
