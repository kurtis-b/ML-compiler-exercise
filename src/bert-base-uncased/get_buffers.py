from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "This is a sample input for BERT model export."
encoded_input = tokenizer(text, return_tensors='pt')

params = dict(model.named_buffers(remove_duplicate=False))

import numpy as np

with open('tensor.csv', 'w') as f:
    for value in params.values():
        if hasattr(value, 'numpy'):
            value = value.numpy()
        np.savetxt(f, [value.flatten()], delimiter=',', fmt='%d')
