#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_openblas

"${CXX:-g++}" -O3 -c mnist_call_benchmark.cpp -o mnist_call_benchmark.o
"${CXX:-g++}" -O3 -no-pie mnist_call_benchmark.o mnist_model_llvm_ir.o -o bench.out \
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_c_runner_utils \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
