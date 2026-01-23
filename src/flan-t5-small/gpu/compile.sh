g++ -c flan_call.cpp -o flan_call.o && g++ flan_call.o flan.o -o a.out \
	-L../../../externals/torch-mlir/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/ -lcuda -lcudart \
	-Wl,-rpath,'./../../../externals/torch-mlir/build/lib' \
	-Wl,-rpath,'/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/'