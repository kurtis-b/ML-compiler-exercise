g++ -O3 -c resnet18_call_benchmark.cpp -o resnet18_call_benchmark.o && g++ -O3 resnet18_call_benchmark.o resnet18.o -o bench.out \
	-L../../../externals/torch-mlir/build/lib -lmlir_c_runner_utils -lmlir_cuda_runtime \
	-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/ -lcuda -lcudart \
	-Wl,-rpath,'./../../../externals/torch-mlir/build/lib' \
	-Wl,-rpath,'/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/'