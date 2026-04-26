#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"

run_gpu=0

usage() {
  cat <<'USAGE'
Usage: bash src/validate_wsl.sh [--gpu]

Runs the supported CPU end-to-end pipelines. Pass --gpu to additionally run
the CUDA-on-WSL pipelines after CPU validation.
USAGE
}

while (($#)); do
  case "$1" in
    --gpu)
      run_gpu=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      pipeline_die "unknown argument: $1"
      ;;
  esac
  shift
done

pipeline_activate_torch_mlir_python
cd "$PIPELINE_REPO_ROOT"

compare_tokens() {
  local actual="$1"
  local expected="$2"
  "$(pipeline_python)" - "$actual" "$expected" <<'PY'
import ast
import re
import sys
from pathlib import Path


def extract_tokens(path):
    text = Path(path).read_text()
    for label in ("Output IDs:", "Generated tokens:"):
        match = re.search(rf"{re.escape(label)}\s*(\[[^\n]+\])", text)
        if match:
            return ast.literal_eval(match.group(1))
    raise SystemExit(f"no token list found in {path}")


actual = extract_tokens(sys.argv[1])
expected = extract_tokens(sys.argv[2])
if actual != expected:
    raise SystemExit(f"token mismatch:\nactual:   {actual}\nexpected: {expected}")
print("Token outputs match")
PY
}

run_sample_cpu() {
  pipeline_note "CPU sample"
  cd "$PIPELINE_REPO_ROOT/src/sample"
  python lower_sample_model.py
  bash run_mlir_pipeline.sh
  python run_sample_model.py > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_mnist_cpu() {
  pipeline_note "CPU mnist"
  cd "$PIPELINE_REPO_ROOT/src/mnist"
  python lower_mnist_model.py
  bash run_mlir_pipeline.sh
  python run_mnist_model.py > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_cnn_cpu() {
  pipeline_note "CPU cnn"
  cd "$PIPELINE_REPO_ROOT/src/cnn"
  python lower_cnn_model.py
  bash run_mlir_pipeline.sh
  python run_cnn_model.py > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_resnet18_cpu() {
  pipeline_note "CPU resnet18"
  cd "$PIPELINE_REPO_ROOT/src/resnet18"
  python lower_resnet18_model.py
  python get_buffers.py
  bash run_mlir_pipeline.sh
  bash compile.sh
  python run_resnet18_model.py > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_bert_cpu() {
  pipeline_note "CPU bert-base-uncased"
  cd "$PIPELINE_REPO_ROOT/src/bert-base-uncased"
  python lower_bert_model.py
  python get_buffers.py
  bash run_mlir_pipeline.sh
  bash compile.sh
  python run_bert.py > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_flan_cpu() {
  pipeline_note "CPU flan-t5-small"
  cd "$PIPELINE_REPO_ROOT/src/flan-t5-small"
  python lower_flan_autoregressive.py
  bash run_mlir_pipeline.sh
  bash compile.sh
  python run_flan_model.py > pytorch_output.txt
  python run_flan_model_mlir.py > mlir_output.txt
  compare_tokens mlir_output.txt pytorch_output.txt
}

run_gpt_cpu() {
  pipeline_note "CPU gpt (${GPT_MODEL_NAME:-sshleifer/tiny-gpt2})"
  cd "$PIPELINE_REPO_ROOT/src/gpt"
  python lower_gpt_model.py
  bash run_mlir_pipeline.sh
  bash compile.sh
  python run_gpt.py > pytorch_output.txt
  python run_gpt_model_mlir.py > mlir_output.txt
  compare_tokens mlir_output.txt pytorch_output.txt
}

run_gpu_numeric_model() {
  local model_dir="$1"
  local lower_script="$2"
  local linalg_file="$3"
  local runner_script="$4"

  pipeline_note "GPU ${model_dir}"
  cd "$PIPELINE_REPO_ROOT/src/${model_dir}"
  python "$lower_script"
  cp "$linalg_file" gpu/
  cd gpu
  bash run_mlir_pipeline.sh
  bash compile.sh
  python "../${runner_script}" > pytorch_output.txt
  ./a.out > mlir_output.txt
  python ../../compare_outputs.py mlir_output.txt pytorch_output.txt
}

run_flan_gpu() {
  pipeline_note "GPU flan-t5-small"
  cd "$PIPELINE_REPO_ROOT/src/flan-t5-small"
  python lower_flan_autoregressive.py
  cp flan_linalg_test.mlir gpu/flan_linalg.mlir
  cd gpu
  bash run_mlir_pipeline.sh
  bash compile.sh
  ./a.out > mlir_output.txt
}

run_sample_cpu
run_mnist_cpu
run_cnn_cpu
run_resnet18_cpu
run_bert_cpu
run_flan_cpu
run_gpt_cpu

if ((run_gpu)); then
  pipeline_require_cuda_tools
  run_gpu_numeric_model sample lower_sample_model.py sample_model_linalg.mlir run_sample_model.py
  run_gpu_numeric_model mnist lower_mnist_model.py mnist_model_linalg.mlir run_mnist_model.py
  run_gpu_numeric_model cnn lower_cnn_model.py cnn_model_linalg.mlir run_cnn_model.py
  run_gpu_numeric_model resnet18 lower_resnet18_model.py resnet18_model_linalg.mlir run_resnet18_model.py
  run_flan_gpu
fi

pipeline_note "WSL validation completed successfully"
