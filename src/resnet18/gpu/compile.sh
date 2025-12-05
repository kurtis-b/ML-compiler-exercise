g++ -c resnet18_call.cpp -o resnet18_call.o && g++ resnet18_call.o resnet18.o -o a.out \
	-L../../../externals/torch-mlir/build/lib -lmlir_runner_utils -lmlir_cuda_runtime \
	-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/ -lcuda -lcudart \
	-Wl,-rpath,'./../../../externals/torch-mlir/build/lib' \
	-Wl,-rpath,'/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/'