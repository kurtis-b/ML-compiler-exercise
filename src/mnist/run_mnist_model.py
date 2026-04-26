import torch
from model import MnistNetwork

# Limit to single thread for reproducibility
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.manual_seed(41)


def print_output():
    model = MnistNetwork()
    input = torch.ones((1, 1, 28, 28))
    output = model(input)
    print(" ".join(f"{value:.5f}" for value in output.flatten().detach().tolist()))

if __name__ == "__main__":
   print_output()
