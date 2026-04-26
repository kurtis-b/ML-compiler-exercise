# Flan-T5 Small

CPU flow:

```bash
python lower_flan_autoregressive.py
bash run_mlir_pipeline.sh
bash compile.sh
python run_flan_model.py > pytorch_output.txt
python run_flan_model_mlir.py > mlir_output.txt
```

`run_flan_model_mlir.py` uses the pybind11 bridge in `flan_call.cpp` to run
greedy token generation through the MLIR-compiled decoder.

GPU flow:

```bash
python lower_flan_autoregressive.py
cp flan_linalg_test.mlir gpu/flan_linalg.mlir
cd gpu
bash run_mlir_pipeline.sh
bash compile.sh
./a.out
```
