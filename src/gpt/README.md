# GPT

The default end-to-end GPT flow uses `sshleifer/tiny-gpt2`, which keeps local
WSL validation practical while still exercising a GPT-2-compatible causal LM.
Set `GPT_MODEL_NAME=gpt2` to try full GPT-2.

CPU flow:

```bash
python lower_gpt_model.py
bash run_mlir_pipeline.sh
bash compile.sh
python run_gpt.py > pytorch_output.txt
python run_gpt_model_mlir.py > mlir_output.txt
```

`gpt_call.cpp` is a pybind11 bridge that calls the MLIR-compiled decoder and
performs greedy token generation.

Useful knobs:

```bash
export GPT_MODEL_NAME=sshleifer/tiny-gpt2
export GPT_PROMPT="What is the capital of France?"
export GPT_MAX_NEW_TOKENS=10
```
