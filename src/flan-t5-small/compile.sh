g++ -O3 -Wall -shared -std=c++17 -fPIC \
	$(python -m pybind11 --includes) \
	flan_call.cpp flan_llvm_test_ir.o \
	-o flan_call$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
	-lm \
	-L../../externals/torch-mlir/build/lib -lmlir_c_runner_utils \
	-L../../openblas/lib -lopenblas \
	-Wl,-rpath=../../externals/torch-mlir/build/lib
