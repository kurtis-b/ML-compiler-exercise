#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

pipeline_require_cuda_tools

ARTIFACT_DIR="${ARTIFACT_DIR:-${PIPELINE_SCRIPT_DIR}/build}"
input="${ARTIFACT_DIR}/04-verified-nvvm-binary.mlir"
pipeline_require_file "$input"

ll_file="${ARTIFACT_DIR}/vector_add.ll"
obj_file="${ARTIFACT_DIR}/vector_add.o"
runner_obj="${ARTIFACT_DIR}/vector_add_runner.o"
runner="${ARTIFACT_DIR}/vector_add_runner"

pipeline_note "translating MLIR to LLVM IR"
"$MLIR_TRANSLATE" -mlir-to-llvmir "$input" -o "$ll_file"

pipeline_note "compiling LLVM IR to an object file"
"$LLC" -filetype=obj -O3 "$ll_file" -o "$obj_file"

cuda_link_flags=(
  -L"${MLIR_RUNNER_LIB_DIR}" -lmlir_runner_utils -lmlir_cuda_runtime
  -L"${CUDA_LIB_DIR}" -L"${CUDA_DRIVER_LIB_DIR}" -lcuda -lcudart
  -Wl,-rpath="${MLIR_RUNNER_LIB_DIR}"
  -Wl,-rpath="${CUDA_LIB_DIR}"
  -Wl,-rpath="${CUDA_DRIVER_LIB_DIR}"
)

pipeline_note "building the C++ host runner"
"${CXX:-g++}" -std=c++17 -O2 -c vector_add_runner.cpp -o "$runner_obj"
"${CXX:-g++}" -no-pie "$runner_obj" "$obj_file" -o "$runner" "${cuda_link_flags[@]}"

pipeline_note "built ${runner}"
