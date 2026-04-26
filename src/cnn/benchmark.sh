#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_openblas

"${CXX:-g++}" -O3 -c cnn_call_benchmark.cpp -o cnn_call_benchmark.o
"${CXX:-g++}" -O3 -no-pie cnn_call_benchmark.o cnn_model_llvm_ir.o -o bench.out \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -lm
