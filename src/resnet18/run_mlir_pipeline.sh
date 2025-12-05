###  Pipeline to get from linalg to llvm dialect  ###

# torch-mlir: Convert from torch dialect to linalg dialect
../../externals/torch-mlir/build/bin/torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline $PWD/resnet18_model_torch.mlir > $PWD/resnet18_model_linalg.mlir

# With BLAS integration (Todo: Can be merged)
../../build-ninja/tools/tutorial-opt --linalg-to-bufferization $PWD/resnet18_model_linalg.mlir > $PWD/resnet18_model_buf_linalg.mlir
../../build-ninja/tools/tutorial-opt --llvm-request-c-wrappers --bufferization-to-llvm $PWD/resnet18_model_buf_linalg.mlir > $PWD/resnet18_model_llvm.mlir

###  Use mlir-translate to get from mlir to llvm ir  ###
mlir-translate -mlir-to-llvmir $PWD/resnet18_model_llvm.mlir > $PWD/resnet18_model_llvm_ir.ll

###  Create .obj file  ###
llc --filetype=obj $PWD/resnet18_model_llvm_ir.ll

###  Compile  ###
g++ -c resnet18_call.cpp -o resnet18_call.o && g++ resnet18_call.o resnet18_model_llvm_ir.o -o a.out \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib