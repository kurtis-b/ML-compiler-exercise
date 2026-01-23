"""
Test the model in native PyTorch to establish expected behavior.
This helps us confirm whether the issue is in the model export/lowering
or in how we're calling the compiled version.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

def test_generation_native():
    """Run generation in native PyTorch to see expected output."""
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.eval()
    
    prompt = "translate English to German: How are you?"
    
    # Encode input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Prompt: {prompt}")
    print(f"Encoded input_ids: {input_ids}")
    print(f"Input shape: {input_ids.shape}")
    print()
    
    # Generate using HuggingFace's built-in generation
    print("=" * 70)
    print("Native HuggingFace generation (greedy):")
    print("=" * 70)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=20,
            num_beams=1,  # greedy
            do_sample=False,
            temperature=1.0
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"Generated tokens: {output_ids[0].tolist()}")
    print(f"Generated text: {generated_text}")
    print()
    
    # Now manually call the model step-by-step to match our C++ approach
    print("=" * 70)
    print("Manual step-by-step generation (matching C++ approach):")
    print("=" * 70)
    
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])
    
    generated_tokens = [tokenizer.pad_token_id]
    
    with torch.no_grad():
        for step in range(15):
            # Pad decoder_input_ids to fixed length
            max_len = 32
            padded = torch.full((1, max_len), tokenizer.pad_token_id, dtype=torch.long)
            decoder_mask = torch.zeros((1, max_len), dtype=torch.long)
            
            current_len = len(generated_tokens)
            padded[0, :current_len] = torch.tensor(generated_tokens)
            decoder_mask[0, :current_len] = 1
            
            print(f"\nStep {step}: seq_len={current_len}")
            print(f"  Generated: {generated_tokens}")
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=padded,
                decoder_attention_mask=decoder_mask,
                return_dict=True
            )
            
            logits = outputs.logits  # shape: (1, max_len, vocab_size)
            print(f"  Logits shape: {logits.shape}")
            
            # Get logits for the last real token position
            # In autoregressive generation, we want logits[0, current_len-1, :]
            # because the model predicts the next token given context up to current_len-1
            last_pos_logits = logits[0, current_len - 1, :]
            
            # Get top 5
            top_5_logits, top_5_ids = torch.topk(last_pos_logits, 5)
            print(f"  Top 5 tokens: {top_5_ids.tolist()}")
            print(f"  Top 5 logits: {top_5_logits.tolist()}")
            
            next_token = top_5_ids[0].item()
            print(f"  → Selected: {next_token}")
            
            if next_token == tokenizer.eos_token_id:
                print("  EOS reached")
                break
            
            generated_tokens.append(next_token)
    
    print("\n" + "=" * 70)
    print(f"Final manual sequence: {generated_tokens}")
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print(f"Decoded: {decoded}")
    print("=" * 70)


if __name__ == "__main__":
    test_generation_native()