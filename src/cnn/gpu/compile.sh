#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_cuda_tools

cuda_link_flags=(
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_runner_utils -lmlir_cuda_runtime
  -L"${CUDA_LIB_DIR}" -L"${CUDA_DRIVER_LIB_DIR}" -lcuda -lcudart
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
  -Wl,-rpath="${CUDA_LIB_DIR}"
  -Wl,-rpath="${CUDA_DRIVER_LIB_DIR}"
)

"${CXX:-g++}" -c cnn_call.cpp -o cnn_call.o
"${CXX:-g++}" -no-pie cnn_call.o cnn.o -o a.out "${cuda_link_flags[@]}" -lm
