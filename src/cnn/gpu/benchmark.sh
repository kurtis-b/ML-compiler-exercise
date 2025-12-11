g++ -O3 -c cnn_call_benchmark.cpp -o cnn_call_benchmark.o && g++ -O3 cnn_call_benchmark.o cnn.o -o bench.out\
	-L../../../externals/torch-mlir/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/ -lcuda -lcudart \
	-lm \
	-Wl,-rpath,'./../../../externals/torch-mlir/build/lib' \
	-Wl,-rpath,'/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/'