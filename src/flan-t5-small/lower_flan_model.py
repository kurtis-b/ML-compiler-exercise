"""
Debug what the exported model actually outputs.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType

class Wrapper(torch.nn.Module):
    def __init__(self, model, max_seq_len=32):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids, attention_mask, decoder_input_ids_padded, decoder_attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids_padded,
            decoder_attention_mask=decoder_attention_mask
        )
        # Return the full logits: (1, seq_len, vocab_size)
        print(f"DEBUG: Model output logits shape: {outputs.logits.shape}")
        return outputs.logits

def test_export():
    MAX_DECODER_LEN = 32
    
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    encoded_input = tokenizer(
        "translate English to German: How are you?",
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()

    wrapped_model = Wrapper(model, max_seq_len=MAX_DECODER_LEN)

    decoder_input_ids_padded = torch.full(
        (1, MAX_DECODER_LEN), 
        tokenizer.pad_token_id, 
        dtype=torch.long
    )
    decoder_attention_mask = torch.zeros((1, MAX_DECODER_LEN), dtype=torch.long)
    decoder_attention_mask[0, 0] = 1

    print("Input shapes:")
    print(f"  input_ids: {encoded_input['input_ids'].shape}")
    print(f"  attention_mask: {encoded_input['attention_mask'].shape}")
    print(f"  decoder_input_ids: {decoder_input_ids_padded.shape}")
    print(f"  decoder_attention_mask: {decoder_attention_mask.shape}")
    print()

    # Test forward pass
    print("Testing forward pass in PyTorch:")
    with torch.no_grad():
        logits = wrapped_model(
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
            decoder_input_ids_padded,
            decoder_attention_mask
        )
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: torch.Size([1, 32, 32128])")
    print()

    # Now export
    print("Exporting model with torch.export...")
    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
            decoder_input_ids_padded,
            decoder_attention_mask
        ),
        dynamic_shapes={
            "input_ids": {0: 1, 1: 10},
            "attention_mask": {0: 1, 1: 10},
            "decoder_input_ids_padded": {0: 1, 1: MAX_DECODER_LEN},
            "decoder_attention_mask": {0: 1, 1: MAX_DECODER_LEN}
        }
    )

    print("ExportedProgram created")
    print(f"Number of graph nodes: {len(list(ep.graph.nodes))}")
    
    # Check the return type
    print("\nExamining exported program output:")
    for node in ep.graph.nodes:
        if node.op == 'output':
            print(f"Output node: {node}")
            print(f"Output args: {node.args}")
    
    ep = ep.run_decompositions()
    
    print("\nConverting to MLIR...")
    m = fx.export_and_import(
        ep,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="transformer_model"
    )

    mlir_str = str(m)
    
    # Parse to find output type
    print("\nSearching for output type in MLIR...")
    for line in mlir_str.split('\n'):
        if 'return' in line or 'func.func' in line or 'memref' in line:
            print(line)
    
    with open("flan_linalg.mlir", "w") as f:
        f.write(mlir_str)
    
    print("\n✓ Model exported to flan_linalg.mlir")

if __name__ == "__main__":
    test_export()