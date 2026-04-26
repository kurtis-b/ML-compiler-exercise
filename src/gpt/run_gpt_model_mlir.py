"""
Run the MLIR-lowered GPT model with greedy decoding for a fixed prompt.
"""

import os

from transformers import AutoTokenizer

import gpt_call

MODEL_NAME = os.environ.get("GPT_MODEL_NAME", "sshleifer/tiny-gpt2")
INPUT_TEXT = os.environ.get("GPT_PROMPT", "What is the capital of France?")
MAX_NEW_TOKENS = int(os.environ.get("GPT_MAX_NEW_TOKENS", "10"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(INPUT_TEXT, return_tensors="pt")

print(f"Model: {MODEL_NAME}")
print(f'Input text: "{INPUT_TEXT}"')
print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])

generated = gpt_call.generate_tokens(
    inputs["input_ids"][0].tolist(),
    inputs["attention_mask"][0].tolist(),
    MAX_NEW_TOKENS,
)

print("Output IDs:", generated)
print("Decoded output:\n", tokenizer.decode(generated, skip_special_tokens=True))
