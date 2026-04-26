# ResNet18

ResNet18 exports batch-normalization running statistics as additional memref
arguments. `get_buffers.py` writes those buffers to `resnet18_buffers.csv`, and
`resnet18_call.cpp` loads them before calling the MLIR C interface.

CPU flow:

```bash
python lower_resnet18_model.py
python get_buffers.py
bash run_mlir_pipeline.sh
bash compile.sh
python run_resnet18_model.py > pytorch_output.txt
./a.out > mlir_output.txt
python ../compare_outputs.py mlir_output.txt pytorch_output.txt
```

GPU flow:

```bash
python lower_resnet18_model.py
cp resnet18_model_linalg.mlir gpu/
cd gpu
bash run_mlir_pipeline.sh
bash compile.sh
./a.out
```
