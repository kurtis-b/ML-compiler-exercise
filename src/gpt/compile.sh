#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_openblas
pybind11_includes="$(pipeline_pybind11_includes)"
python_ext_suffix="$(pipeline_python_ext_suffix)"

"${CXX:-g++}" -O3 -Wall -shared -std=c++17 -fPIC \
  ${pybind11_includes} \
  gpt_call.cpp gpt_llvm_ir.o \
  -o "gpt_call${python_ext_suffix}" \
  -lm \
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_c_runner_utils \
  -L"${OPENBLAS_LIB_DIR}" -lopenblas \
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
