#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_torch_mlir_tools

if [[ -f resnet18_model_linalg.mlir ]]; then
  input_mlir="resnet18_model_linalg.mlir"
elif [[ -f resnet18_model_torch.mlir ]]; then
  "$TORCH_MLIR_OPT" -torch-backend-to-linalg-on-tensors-backend-pipeline resnet18_model_torch.mlir > resnet18_model_linalg.mlir
  input_mlir="resnet18_model_linalg.mlir"
else
  pipeline_die "expected resnet18_model_linalg.mlir or resnet18_model_torch.mlir in ${PIPELINE_SCRIPT_DIR}"
fi

"$TUTORIAL_OPT" -linalg-to-bufferization "$input_mlir" > resnet18_buf_linalg.mlir
"$TUTORIAL_OPT" -llvm-request-c-wrappers -bufferization-to-llvm resnet18_buf_linalg.mlir > resnet18_llvm.mlir

"$MLIR_TRANSLATE" -mlir-to-llvmir resnet18_llvm.mlir | "$LLC" --filetype=obj -O3 -o resnet18_llvm_ir.o
