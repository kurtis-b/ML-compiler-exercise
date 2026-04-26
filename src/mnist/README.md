# MNIST

CPU flow:

```bash
python lower_mnist_model.py
bash run_mlir_pipeline.sh
python run_mnist_model.py > pytorch_output.txt
./a.out > mlir_output.txt
python ../compare_outputs.py mlir_output.txt pytorch_output.txt
```

GPU flow:

```bash
python lower_mnist_model.py
cp mnist_model_linalg.mlir gpu/
cd gpu
bash run_mlir_pipeline.sh
bash compile.sh
./a.out
```
