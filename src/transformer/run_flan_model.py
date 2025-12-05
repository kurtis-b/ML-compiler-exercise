"""
    Print the output of a sample model given an example text input.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How are you?"
input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids

print(f'Input text: "{input_text}"')
print("Corresponding input IDs: ", input_ids)

# Run the model to get output IDs
outputs = model.generate(input_ids)

print("Output IDs: ", outputs)
print("Decoded output: ", tokenizer.decode(outputs[0]))