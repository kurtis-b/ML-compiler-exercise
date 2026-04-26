#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_openblas

"${CXX:-clang++}" -c call_bert.cpp -o call_bert.o
"${CXX:-clang++}" -no-pie -lm call_bert.o bert_llvm_ir.o -o a.out \
  -L"${MLIR_RUNNER_LIB_DIR}" \
  -L"${OPENBLAS_LIB_DIR}" \
  -lmlir_c_runner_utils \
  -lopenblas \
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
