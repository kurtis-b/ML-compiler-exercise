#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_common_tools
pipeline_require_openblas

"$TUTORIAL_OPT" --linalg-to-bufferization "$PIPELINE_SCRIPT_DIR/mnist_model_linalg.mlir" > "$PIPELINE_SCRIPT_DIR/mnist_model_buf_linalg.mlir"
"$TUTORIAL_OPT" --llvm-request-c-wrappers --bufferization-to-llvm "$PIPELINE_SCRIPT_DIR/mnist_model_buf_linalg.mlir" > "$PIPELINE_SCRIPT_DIR/mnist_model_llvm.mlir"

"$MLIR_TRANSLATE" -mlir-to-llvmir "$PIPELINE_SCRIPT_DIR/mnist_model_llvm.mlir" > "$PIPELINE_SCRIPT_DIR/mnist_model_llvm_ir.ll"

"$LLC" --filetype=obj "$PIPELINE_SCRIPT_DIR/mnist_model_llvm_ir.ll" -o "$PIPELINE_SCRIPT_DIR/mnist_model_llvm_ir.o"

"${CXX:-g++}" -c mnist_call.cpp -o mnist_call.o
"${CXX:-g++}" -no-pie mnist_call.o mnist_model_llvm_ir.o -o a.out \
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_c_runner_utils \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
