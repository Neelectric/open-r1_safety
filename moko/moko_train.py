# Multi-Objective KL Optimization (MOKO)


from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, set_seed, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


set_seed(
    42, 
    # deterministic=True # this refuses to work: RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    )

### Model, Tokenizer, hyperparams prep
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
max_length = 2048
batch_size = 8
num_epochs = 3


### Dataset prep
dataset_raw = load_dataset("Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks")["train"]
dataset_subset = dataset_raw.select(range(0,100))


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
        truncation=True, 
        padding='max_length',
        max_length=max_length
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

### Data Collator and Data Loader prep
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
for batch in train_dataloader:
    break
# batched = {k: v.shape for k, v in batch.items()}
# print(batched)

# outputs = model(**batch)
# print(outputs)

### Optimizer and LR scheduler prep
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps // 20,
    num_training_steps=num_training_steps,
)
print(f"Training for {num_training_steps} step")
device = model.device
progress_bar = tqdm(range(num_training_steps))


### training!
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        tqdm.write(f"loss is {loss}")