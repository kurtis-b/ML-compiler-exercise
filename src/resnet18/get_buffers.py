from transformers import AutoModelForImageClassification


model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
model.eval()

with open("resnet18_buffers.csv", "w") as f:
    for _, value in model.named_buffers(remove_duplicate=False):
        tensor = value.detach().cpu()
        if tensor.dim() == 0:
            f.write(f"i,{int(tensor.item())}\n")
            continue

        flattened = tensor.reshape(-1).tolist()
        serialized = ",".join(f"{element:.9g}" for element in flattened)
        f.write(f"f,{serialized}\n")
