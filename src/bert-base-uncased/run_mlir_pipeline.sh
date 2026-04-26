#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_torch_mlir_tools

if [[ -f bert_torch.mlir ]]; then
  source_ir="bert_torch.mlir"
elif [[ -f bert_torch_both.mlir ]]; then
  source_ir="bert_torch_both.mlir"
else
  pipeline_die "expected bert_torch.mlir in ${PIPELINE_SCRIPT_DIR}"
fi

"$TORCH_MLIR_OPT" -torch-backend-to-linalg-on-tensors-backend-pipeline "$source_ir" > bert_linalg.mlir
"$TUTORIAL_OPT" -linalg-to-bufferization bert_linalg.mlir > bert_buf_linalg.mlir
"$TUTORIAL_OPT" -llvm-request-c-wrappers -bufferization-to-llvm bert_buf_linalg.mlir > bert_llvm.mlir
"$MLIR_TRANSLATE" -mlir-to-llvmir bert_llvm.mlir | "$LLC" --filetype=obj -O3 -o bert_llvm_ir.o
