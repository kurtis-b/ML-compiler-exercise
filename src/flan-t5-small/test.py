from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed

# Reproducibility
set_seed(42)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "translate English to German: How are you?"

inputs = tokenizer(text, return_tensors="pt")

print(inputs["input_ids"])
print(inputs["attention_mask"])


outputs = model.generate(
    inputs["input_ids"],
    #max_length=40,
    #num_beams=5,
    #early_stopping=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Generated token IDs: ", outputs[0].tolist())