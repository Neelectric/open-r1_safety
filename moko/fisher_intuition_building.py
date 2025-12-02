# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, set_seed, get_scheduler, AutoConfig
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

# claude suggested this approach:
from accelerate import infer_auto_device_map, init_empty_weights

set_seed(
    42, 
    # deterministic=True # this refuses to work:
    )

### Model, Tokenizer, hyperparams prep
# model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model_id = "allenai/OLMo-2-0425-1B-Instruct"
model_id = "allenai/OLMo-2-1124-13B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

#claude's way
# config = AutoConfig.from_pretrained(model_id)

# with init_empty_weights():
#     empty_model = AutoModelForCausalLM.from_config(config)

# # Create balanced map across all 8 GPUs
# device_map = infer_auto_device_map(
#     empty_model,
#     # max_memory={i: "90GiB" for i in range(8)},  # leave headroom
#     no_split_module_classes=["Olmo2DecoderLayer"],
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     dtype=torch.bfloat16,
#     device_map=device_map,
# )

#my way
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype="bfloat16",
    device_map="balanced",
    # attn_implementation="flash_attention_2",
    )



max_length = 4096
batch_size = 1
num_epochs = 1

# %%
### Dataset prep
# dataset_id = "Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks"
dataset_id = "Neelectric/OpenR1-Math-220k_CN-K12_OLMo-2_4096toks"

dataset_raw = load_dataset(dataset_id)["train"]
dataset_raw = dataset_raw.shuffle()
dataset_subset = dataset_raw.select(range(0,2500))

# %%
### Dataset tokenization
system_prompt_content = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
system_prompt = [{
    "content": system_prompt_content,
    "role": "system"
    }]

def tokenize_function(example):
    messages =  system_prompt + example["messages"]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(
        templated, 
        # return_tensors="pt", #it appears this causes addition of redundant batch dimension, e.g.
        # {'input_ids': torch.Size([8, 1, 2048]), 'attention_mask': torch.Size([8, 1, 2048])} instead of {'input_ids': torch.Size([8, 2048]), 'attention_mask': torch.Size([8, 2048])}
        
        # #old vals
        # truncation=True, 
        # padding='max_length',
        # max_length=max_length
        
        # new vals
        truncation=False, 
        padding=False,
        )
    labels = tokenized["input_ids"].copy()
    # we need to be careful to mask padding tokens!!!
    labels = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels]
    tokenized["labels"] = labels
    return tokenized


first_sample = dataset_subset[0]
tokenized = tokenize_function(first_sample)
tokenized_dataset = dataset_subset.map(tokenize_function, num_proc=16)
tokenized_dataset = tokenized_dataset.remove_columns(['problem', 'solution', 'answer', 'problem_type', 'question_type', 'source', 'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 'finish_reasons', 'correctness_count', 'messages'])
print(tokenized_dataset)

# %%
### Data Collator and Data Loader prep
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
for batch in train_dataloader:
    break
# batched = {k: v.shape for k, v in batch.items()}
# print(batched)
# outputs = model(**batch.to(model.device))
# print(outputs)
batch["input_ids"]

# %%
tokenizer.decode(batch["input_ids"][0])

# %% [markdown]
# # Let's compute the Fisher Diagonal

# %%
batch["input_ids"].shape

# %%
print(len(train_dataloader))

# %%
num_training_steps = num_epochs * len(train_dataloader)
print(f"Training for {num_training_steps} steps")
device = model.device
progress_bar = tqdm(range(num_training_steps))

from copy import deepcopy
import torch


fisher = {}
params = {name: param for name, param in model.named_parameters() if param.requires_grad}
for name, param in deepcopy(params).items():
    param.data.zero_()
    fisher[name] = param.data

### fisher estimation loop
model.eval()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        # print(outputs)
        progress_bar.update(1)
        tqdm.write(f"loss is {loss.item()}")
        for name, param in model.named_parameters():
            fisher[name].data += param.grad.data ** 2 / len(train_dataloader)
    break

# %%
params['model.embed_tokens.weight'][0]

# %%
fisher['model.embed_tokens.weight'][0]

# %%


# %% [markdown]
# # let's try and plot this?

# %%


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Adjust based on your model
num_layers = len(model.model.layers)

# Map display names to actual parameter substrings
param_map = {
    'Queries': 'q_proj',
    'Keys': 'k_proj', 
    'Values': 'v_proj',
    'Outputs': 'o_proj',
    'Gate': 'gate_proj',
    'Up': 'up_proj',
    'Down': 'down_proj'
}
param_types = list(param_map.keys())

# Create matrix to store mean FIM values
fim_matrix = np.zeros((num_layers, len(param_types)))



# Populate the matrix
for name, fim_values in fisher.items():
    for i, ptype in enumerate(param_types):
        if param_map[ptype] in name:  # Match against actual param names
            parts = name.split('.')
            for j, part in enumerate(parts):
                if part == 'layers' and j + 1 < len(parts):
                    layer_num = int(parts[j + 1])
                    fim_matrix[layer_num, i] = fim_values.mean().item()
                    break
            break

# Create DataFrame
df = pd.DataFrame(fim_matrix, 
                  index=[f'Layer {i}' for i in range(num_layers)],
                  columns=param_types)

# Reverse so Layer 31 is at top
df = df.iloc[::-1]

# Plot
plt.figure(figsize=(10, 12))
sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Mean FIM'})
plt.xlabel('Parameters')
plt.ylabel('Layers')
plt.title('Mean Fisher Information per matrix:\nOLMo-2-1124-7B-Instruct')
plt.tight_layout()
plt.savefig('fim_heatmap.png', dpi=150)
plt.show()


