g++ -O3 -c mnist_call_benchmark.cpp -o mnist_call_benchmark.o && g++ mnist_call_benchmark.o mnist_model_llvm_ir.o -o bench.out\
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib