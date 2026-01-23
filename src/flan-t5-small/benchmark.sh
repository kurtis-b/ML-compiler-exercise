g++ -O3 -c flan_call_benchmark.cpp -o flan_call_benchmark.o && g++ flan_call_benchmark.o flan_llvm_ir.o -o bench.out\
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../openblas/lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib