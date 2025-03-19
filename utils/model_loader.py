import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

qconf = BitsAndBytesConfig(load_in_8bit=True)
model_name = "AnatoliiPotapov/T-lite-instruct-0.1"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device_map = "auto:0"
else:
    device_map = None
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map=device_map, torch_dtype="auto", quantization_config=qconf
)
if device_map is None:
    model = model.to(device)
print(f"model {model_name.split('/')[1]} successully loaded on {model.device}")
