###  Pipeline to get from linalg to llvm dialect  ###

# For Blas integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/flan_linalg.mlir > $PWD/flan_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/flan_buf_linalg.mlir > $PWD/flan_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/flan_llvm.mlir > $PWD/flan_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/flan_llvm_ir.ll

###  Compile  ###
g++ -c flan_call.cpp -o flan_call.o && g++ flan_call.o flan_llvm_ir.o -o a.out \
	-lm \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib 