from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
text = "This is a sample input for BERT model export."
encoded_input = tokenizer(text, return_tensors="pt")

output = model(**encoded_input)

for row in output.last_hidden_state[0, :, :10].detach().tolist():
    print(" ".join(f"{value:.5f}" for value in row))

print()
print(" ".join(f"{value:.5f}" for value in output.pooler_output[0, :10].detach().tolist()))
