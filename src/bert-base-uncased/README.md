# BERT Base Uncased

CPU flow:

```bash
python lower_bert_model.py
python get_buffers.py
bash run_mlir_pipeline.sh
bash compile.sh
python run_bert.py > pytorch_output.txt
./a.out > mlir_output.txt
python ../compare_outputs.py mlir_output.txt pytorch_output.txt
```

The runner compares the MLIR-compiled model output against the PyTorch reference
for a fixed tokenizer input.
