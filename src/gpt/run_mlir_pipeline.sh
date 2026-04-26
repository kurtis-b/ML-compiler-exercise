#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_torch_mlir_tools
pipeline_require_file gpt_torch.mlir

"$TORCH_MLIR_OPT" -torch-backend-to-linalg-on-tensors-backend-pipeline gpt_torch.mlir > gpt_linalg.mlir
"$TUTORIAL_OPT" -linalg-to-bufferization gpt_linalg.mlir > gpt_buf_linalg.mlir
"$TUTORIAL_OPT" -llvm-request-c-wrappers -bufferization-to-llvm gpt_buf_linalg.mlir > gpt_llvm.mlir
"$MLIR_TRANSLATE" -mlir-to-llvmir gpt_llvm.mlir | "$LLC" --filetype=obj -O3 -relocation-model=pic -o gpt_llvm_ir.o
