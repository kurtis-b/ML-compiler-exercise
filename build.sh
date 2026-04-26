#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/src/pipeline_common.sh"

cd "$PIPELINE_REPO_ROOT"

BUILD_SYSTEM="${BUILD_SYSTEM:-Ninja}"
: "${BUILD_DIR:=${PIPELINE_REPO_ROOT}/build-ninja}"
: "${CMAKE_BUILD_TYPE:=Debug}"
: "${BUILD_DEPS:=ON}"
: "${BUILD_SHARED_LIBS:=OFF}"

LLVM_DIR_DEFAULT="${TORCH_MLIR_BUILD_DIR}/lib/cmake/llvm"
MLIR_DIR_DEFAULT="${TORCH_MLIR_BUILD_DIR}/lib/cmake/mlir"

: "${LLVM_DIR:=${LLVM_DIR_DEFAULT}}"
: "${MLIR_DIR:=${MLIR_DIR_DEFAULT}}"

pipeline_require_dir "$TORCH_MLIR_BUILD_DIR"
pipeline_require_dir "$LLVM_DIR"
pipeline_require_dir "$MLIR_DIR"

targets=(mlir-headers mlir-doc tutorial-opt)
if [[ $# -gt 0 ]]; then
  targets=("$@")
fi

cmake -S "$PIPELINE_REPO_ROOT" -B "$BUILD_DIR" -G "$BUILD_SYSTEM" \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR" \
  -DTORCH_MLIR_BUILD_DIR="$TORCH_MLIR_BUILD_DIR" \
  -DBUILD_DEPS="$BUILD_DEPS" \
  -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"

build_args=()
if [[ -n "${BUILD_JOBS:-}" ]]; then
  build_args+=(--parallel "$BUILD_JOBS")
fi

for target in "${targets[@]}"; do
  cmake --build "$BUILD_DIR" "${build_args[@]}" --target "$target"
done
