import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_gpu_memory():
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: Free = {free:.2f} GB | Total = {total:.2f} GB")
    else:
        print("CUDA not available")


torch.cuda.empty_cache()
model_name = "AnatoliiPotapov/T-lite-instruct-0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device_map = "auto"
else:
    device_map = None
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.float16)
if device_map is None:
    model = model.to(device)
print(f"model {model_name.split('/')[1]} successully loaded on {model.device}")

print_gpu_memory()
