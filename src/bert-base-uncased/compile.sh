#!/usr/bin/env bash

set -euo pipefail

"${CXX:-clang++}" -c call_bert.cpp -o call_bert.o
"${CXX:-clang++}" -no-pie -lm call_bert.o bert_llvm_ir.o -o a.out \
    -L../../externals/torch-mlir/build/lib \
    -L../../openblas/lib \
    -lmlir_c_runner_utils \
    -lopenblas \
    -Wl,-rpath=../../externals/torch-mlir/build/lib
