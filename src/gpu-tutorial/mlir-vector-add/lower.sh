#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_cuda_tools

ARTIFACT_DIR="${ARTIFACT_DIR:-${PIPELINE_SCRIPT_DIR}/build}"
mkdir -p "$ARTIFACT_DIR"

input="vector_add_gpu.mlir"
verified="${ARTIFACT_DIR}/01-verified.mlir"
outlined="${ARTIFACT_DIR}/02-outlined.mlir"
lowered="${ARTIFACT_DIR}/03-nvvm-binary.mlir"
final_verified="${ARTIFACT_DIR}/04-verified-nvvm-binary.mlir"

pipeline_note "verifying hand-written MLIR GPU input"
"$MLIR_OPT" "$input" --verify-each -o "$verified"

pipeline_note "outlining gpu.launch into a gpu.module/gpu.func"
"$MLIR_OPT" "$verified" \
  --gpu-kernel-outlining \
  --canonicalize \
  --cse \
  -o "$outlined"

pipeline_note "lowering outlined GPU code to NVVM and serializing a GPU binary"
"$MLIR_OPT" "$outlined" \
  --llvm-request-c-wrappers \
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
  --gpu-to-llvm='use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true' \
  --gpu-module-to-binary \
  -o "$lowered"

pipeline_note "verifying final GPU binary-bearing MLIR"
"$MLIR_OPT" "$lowered" --verify-each -o "$final_verified"

pipeline_note "wrote MLIR artifacts to ${ARTIFACT_DIR}"
