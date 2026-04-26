"""
Export a GPT-2-compatible causal language model to torch-mlir.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType

MODEL_NAME = os.environ.get("GPT_MODEL_NAME", "sshleifer/tiny-gpt2")
INPUT_TEXT = os.environ.get("GPT_PROMPT", "What is the capital of France?")


class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model, return_last_logits_only=False):
        super().__init__()
        self.model = model
        self.return_last_logits_only = return_last_logits_only

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        if not self.return_last_logits_only:
            return logits
        return logits[:, -1:, :]


def export_gpt2_for_autoregressive(return_last_only=True):
    print("=" * 70)
    print("Exporting GPT for Autoregressive Generation")
    print("=" * 70)

    print("\n1. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    encoded_input = tokenizer(
        INPUT_TEXT,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    print(f"   Model: {MODEL_NAME}")
    print(f"   Input text: '{INPUT_TEXT}'")
    print(f"   input_ids shape: {encoded_input['input_ids'].shape}")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.config.use_cache = False

    print("   Model config:")
    print(f"     - hidden_size: {model.config.hidden_size}")
    print(f"     - num_layers: {model.config.num_hidden_layers}")
    print(f"     - num_attention_heads: {model.config.num_attention_heads}")

    print("\n2. Warming up model...")
    with torch.no_grad():
        _ = model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
        )
    print("   Warm-up complete")

    print("\n3. Creating wrapper...")
    wrapped_model = GPT2Wrapper(model, return_last_logits_only=return_last_only)

    print("\n4. Setting up dynamic sequence length dimension...")
    seq_len_dim = torch.export.Dim("seq_len", max=128)
    print("   Sequence can be: 1 to 128 tokens")

    print("\n5. Exporting with torch.export.export()...")
    print(f"   Input shapes:")
    print(f"     - input_ids: {encoded_input['input_ids'].shape}")
    print(f"     - attention_mask: {encoded_input['attention_mask'].shape}")

    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
        ),
        dynamic_shapes={
            "input_ids": {1: seq_len_dim},
            "attention_mask": {1: seq_len_dim},
        },
    )
    print("   Export succeeded")

    print("\n6. Running decompositions...")
    ep = ep.run_decompositions()
    print("   Decompositions complete")

    print("\n7. Importing to MLIR...")
    m = fx.export_and_import(
        ep,
        output_type=OutputType.TORCH,
        func_name="gpt2_decoder",
    )
    print("   MLIR import succeeded")

    print("\n8. Saving MLIR to file...")
    mlir_str = str(m)
    output_file = "gpt_torch.mlir"
    with open(output_file, "w") as f:
        f.write(mlir_str)

    print(f"   MLIR written to {output_file}")
    print(f"   File size: {len(mlir_str):,} bytes")

    return wrapped_model, encoded_input, tokenizer, ep


if __name__ == "__main__":
    print("\nGPT Export for Autoregressive Generation")
    print("=" * 70)

    try:
        export_gpt2_for_autoregressive(return_last_only=True)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            """
Export successful
MLIR file: gpt_torch.mlir
Supports dynamic sequence length (1-128 tokens)

For autoregressive generation:
1. Export once with dynamic sequence length
2. Each inference call:
   - input_ids: (batch, 1..128) grows with each token
   - attention_mask: (batch, 1..128) grows with the same length
3. Extract logits[:, -1, :] to get the next token
4. Append the token and repeat
"""
        )

    except Exception as e:
        print(f"\nError: {type(e).__name__}")
        print(f"  {str(e)}")
        import traceback

        traceback.print_exc()
