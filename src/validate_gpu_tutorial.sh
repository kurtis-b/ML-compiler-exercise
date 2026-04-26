#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"
cd "$PIPELINE_REPO_ROOT"

size="${GPU_TUTORIAL_SIZE:-1024}"

pipeline_note "GPU tutorial validation uses tiny inputs only (GPU_TUTORIAL_SIZE=${size})"

if pipeline_cuda_smoke >/dev/null 2>&1; then
  pipeline_note "CUDA visibility: $(pipeline_cuda_smoke | head -n1)"
else
  pipeline_skip "nvidia-smi is not available; CUDA-backed tutorial lanes will be skipped"
fi

if [[ -f "$MLIR_OPT" && -f "$MLIR_TRANSLATE" && -f "$LLC" ]] && (pipeline_require_cuda_tools >/dev/null 2>&1); then
  pipeline_note "running MLIR GPU vector-add tutorial"
  (cd src/gpu-tutorial/mlir-vector-add && GPU_TUTORIAL_SIZE="$size" bash run.sh)
else
  pipeline_skip "MLIR GPU vector add requires mlir-opt, mlir-translate, llc, and libmlir_cuda_runtime from a CUDA-enabled torch-mlir build"
fi

python_bin="$(pipeline_python)"
triton_ready_output="$("$python_bin" - <<'PY' 2>&1 || true
try:
    import torch
    import triton
except Exception as exc:
    print(exc)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is false")
    raise SystemExit(1)
print(f"torch={torch.__version__} triton={triton.__version__}")
PY
)"

if [[ "$triton_ready_output" == torch=* ]]; then
  pipeline_note "Triton readiness: ${triton_ready_output}"
  pipeline_prepare_triton_dirs
  "$python_bin" src/gpu-tutorial/triton-vector-add/vector_add.py --size "$size"
  "$python_bin" src/gpu-tutorial/triton-softmax/softmax.py --rows 4 --cols 32
  if [[ "${GPU_TUTORIAL_DUMP_IR:-0}" == "1" ]]; then
    "$python_bin" src/gpu-tutorial/triton-vector-add/vector_add.py --size "$size" --dump-ir
  else
    pipeline_note "set GPU_TUTORIAL_DUMP_IR=1 to also smoke-test Triton IR dumping"
  fi
else
  pipeline_skip "Triton examples require PyTorch, Triton, and torch.cuda; readiness check said: ${triton_ready_output}"
fi

pipeline_note "GPU tutorial validation completed"
