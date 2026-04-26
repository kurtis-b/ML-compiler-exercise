#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_cuda_tools

"$MLIR_OPT" resnet18_model_linalg.mlir \
  --convert-tensor-to-linalg \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --buffer-deallocation-pipeline \
  --convert-bufferization-to-memref \
  --llvm-request-c-wrappers \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --canonicalize \
  --cse \
| "$MLIR_OPT" \
  --pass-pipeline='builtin.module(func.func(affine-loop-invariant-code-motion))' \
  --pass-pipeline='builtin.module(func.func(convert-affine-for-to-gpu))' \
| "$MLIR_OPT" \
  --gpu-kernel-outlining \
  --lower-affine \
  --gpu-decompose-memrefs \
  --expand-strided-metadata \
  --normalize-memrefs \
  --convert-index-to-llvm \
  --arith-expand \
  --memref-expand \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_${MLIR_CUDA_ARCH} opt-level=3" \
  --convert-nvvm-to-llvm \
  --reconcile-unrealized-casts \
  --gpu-to-llvm='use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false' \
  --gpu-module-to-binary \
  -o resnet18_nvvm.mlir

"$MLIR_TRANSLATE" -mlir-to-llvmir resnet18_nvvm.mlir -o resnet18.ll
"$LLC" -filetype=obj -O3 resnet18.ll -o resnet18.o
