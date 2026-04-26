#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Triton vector-add tutorial")
    parser.add_argument("--size", type=int, default=int(os.environ.get("GPU_TUTORIAL_SIZE", "1024")))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dump-ir", action="store_true", help="dump Triton compiler IR to repo-local folders")
    return parser.parse_args()


def configure_dumping(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    dump_dir = repo_root / ".triton-dumps" / "vector-add"
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


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def list_dumped_files(path: Path) -> None:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        print(f"No Triton dump files were observed under {path}")
        return
    print(f"Triton dump files under {path}:")
    for file in files:
        print(f"  {file.relative_to(path)}")


def main() -> None:
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false; this example needs a CUDA-visible PyTorch install")

    torch.manual_seed(0)
    x = torch.arange(args.size, device="cuda", dtype=torch.float32) * 0.5
    y = (torch.arange(args.size, device="cuda", dtype=torch.float32) % 17) - 3.0
    out = torch.empty_like(x)

    grid = (triton.cdiv(args.size, args.block_size),)
    vector_add_kernel[grid](x, y, out, args.size, BLOCK_SIZE=args.block_size)

    expected = x + y
    max_abs_error = torch.max(torch.abs(out - expected)).item()
    if not torch.allclose(out, expected, atol=1.0e-6, rtol=0.0):
        raise SystemExit(f"vector add mismatch: max_abs_error={max_abs_error:.8g}")

    print(f"Triton vector add n={args.size} max_abs_error={max_abs_error:.8g}")
    if args.dump_ir:
        list_dumped_files(dump_dir)
        print(f"Triton cache directory: {cache_dir}")


if __name__ == "__main__":
    main()
