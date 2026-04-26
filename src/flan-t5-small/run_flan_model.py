from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: What is your name?"
input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

print(f'Input text: "{input_text}"')
print("Corresponding input IDs: ", input.input_ids)
print("Corresponding attention: ", input.attention_mask)

outputs = model.generate(
    input.input_ids,
    attention_mask=input.attention_mask,
    do_sample=False,
    max_new_tokens=20,
)

print("Output IDs: ", outputs[0].tolist())
print("Decoded output: ", tokenizer.decode(outputs[0], skip_special_tokens=True))
