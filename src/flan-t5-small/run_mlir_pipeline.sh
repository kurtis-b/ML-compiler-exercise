#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_common_tools

if [[ -f flan_linalg_test.mlir ]]; then
  input_mlir="flan_linalg_test.mlir"
elif [[ -f flan_linalg.mlir ]]; then
  input_mlir="flan_linalg.mlir"
else
  pipeline_die "expected flan_linalg_test.mlir or flan_linalg.mlir in ${PIPELINE_SCRIPT_DIR}"
fi

"$TUTORIAL_OPT" -linalg-to-bufferization "$input_mlir" > flan_buf_linalg.mlir
"$TUTORIAL_OPT" -llvm-request-c-wrappers -bufferization-to-llvm flan_buf_linalg.mlir > flan_llvm.mlir
"$MLIR_TRANSLATE" -mlir-to-llvmir flan_llvm.mlir | "$LLC" --filetype=obj -O3 -relocation-model=pic -o flan_llvm_test_ir.o
