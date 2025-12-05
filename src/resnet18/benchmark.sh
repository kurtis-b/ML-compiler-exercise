g++ -O3 -c resnet18_call_benchmark.cpp -o resnet18_call_benchmark.o && g++ resnet18_call_benchmark.o resnet18_model_llvm_ir.o -o bench.out \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib