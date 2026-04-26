#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_common_tools
pipeline_require_openblas

"$TUTORIAL_OPT" --linalg-to-bufferization "$PIPELINE_SCRIPT_DIR/sample_model_linalg.mlir" > "$PIPELINE_SCRIPT_DIR/sample_model_buf_linalg.mlir"
"$TUTORIAL_OPT" --llvm-request-c-wrappers --bufferization-to-llvm "$PIPELINE_SCRIPT_DIR/sample_model_buf_linalg.mlir" > "$PIPELINE_SCRIPT_DIR/sample_model_llvm.mlir"

"$MLIR_TRANSLATE" -mlir-to-llvmir "$PIPELINE_SCRIPT_DIR/sample_model_llvm.mlir" > "$PIPELINE_SCRIPT_DIR/sample_model_llvm_ir.ll"

"$LLC" --filetype=obj "$PIPELINE_SCRIPT_DIR/sample_model_llvm_ir.ll" -o "$PIPELINE_SCRIPT_DIR/sample_model_llvm_ir.o"

"${CXX:-g++}" -c sample_call.cpp -o sample_call.o
"${CXX:-g++}" -no-pie sample_call.o sample_model_llvm_ir.o -o a.out \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -lm
