"""
    Decode a sequence of token IDs using the T5 tokenizer.
	The sequence is read from 'final_sequence.txt' which is generated 
	by the compiled model and called from flan_call.cpp.
"""

from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

with open('final_sequence.txt', 'r') as f:
	sequence = f.read().strip().split()
	sequence = [int(x) for x in sequence]

print("Input sequence: ", sequence)
# Decode the generated token ID sequence
print(tokenizer.decode(sequence))