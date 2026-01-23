###  Pipeline to get from linalg to llvm dialect  ###

# For Blas integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt t5_linalg.mlir --linalg-to-bufferization \
| ../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm -o t5_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir t5_llvm.mlir -o t5_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj -O3 t5_llvm_ir.ll

###  Compile  ###
g++ -c flan_call.cpp -o flan_call.o && g++ flan_call.o t5_llvm_ir.o -o a.out \
	-lm \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../openblas/lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib 