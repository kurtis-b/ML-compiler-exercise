g++ -c resnet18_call.cpp -o resnet18_call.o && g++ resnet18_call.o resnet18_llvm_ir.o -o a.out \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../openblas/lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib