#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_openblas

"${CXX:-g++}" -O3 -c flan_call_benchmark.cpp -o flan_call_benchmark.o
"${CXX:-g++}" -O3 -no-pie flan_call_benchmark.o flan_llvm_test_ir.o -o bench.out \
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_c_runner_utils \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
