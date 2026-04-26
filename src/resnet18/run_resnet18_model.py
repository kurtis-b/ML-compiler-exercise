import torch

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)

from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
model.eval()

def print_output():
    x = torch.ones((1, 3, 224, 224))
    output = model(x)
    print(" ".join(f"{value:.5f}" for value in output.logits[0].detach().tolist()))

if __name__ == "__main__":
   print_output()
