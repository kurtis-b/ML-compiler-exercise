g++ -c cnn_call.cpp -o cnn_call.o && g++ cnn_call.o cnn.o -o a.out \
	-lm \
	-L/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/ -lcuda -lcudart \
	-Wl,-rpath,'./../../../externals/torch-mlir/build/lib' \
	-Wl,-rpath,'/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.6.3/lib64/'