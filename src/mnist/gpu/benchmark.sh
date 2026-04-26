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

"${CXX:-g++}" -O3 -c mnist_call_benchmark.cpp -o mnist_call_benchmark.o
"${CXX:-g++}" -O3 -no-pie mnist_call_benchmark.o mnist.o -o bench.out "${cuda_link_flags[@]}"
