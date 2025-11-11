from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "allenai/OLMo-2-0425-1B-Instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="cpu"  # Keep on CPU for conversion
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save in fp16
output_path = "/home/user/repos/open-r1_safety/models_in_fp16/" + model_id
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)