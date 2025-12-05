from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch_mlir import fx
from torch_mlir.fx import OutputType

""" A simple wrapper around the Flan-T5-small model to adapt its forward method for export."""
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        return outputs.logits   # decoder output distribution


""" Exports the Flan-T5-small model to Linalg on Tensors dialect in MLIR."""
def export_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    encoded_input = tokenizer("translate English to German: How are you?",
                          return_tensors="pt", padding=True, truncation=True)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    wrapped_model = Wrapper(model)

    # Start the decoder with just the <pad> token
    decoder_start = torch.tensor([[tokenizer.pad_token_id, tokenizer.pad_token_id]])
    dec_len = torch.export.Dim("dec_len", max=16)

    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
            decoder_start
        ),
        dynamic_shapes={
            "input_ids": {0: 1, 1: 10},             # input_ids
            "attention_mask": {0: 1, 1: 10},        # attention_mask (same dim)
            "decoder_input_ids": {0: 1, 1: dec_len} # decoder_input_ids
        }
    )

    ep = ep.run_decompositions()
    m = fx.export_and_import(
        ep,
        output_type=OutputType.LINALG_ON_TENSORS,
        func_name="transformer_model"
    )

    mlir_str = str(m)
    with open("flan_linalg.mlir", "w") as f:
        f.write(mlir_str)

if __name__ == "__main__":
    export_model()