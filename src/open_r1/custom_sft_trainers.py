

import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.distributed as dist

from trl import ModelConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from transformers.optimization import get_constant_schedule_with_warmup
from transformers import TrainerCallback
from datasets import load_dataset
import accelerate

from tqdm import tqdm

from open_r1.optims.dadamw import DAdamW, setup_dadamw


class SFTTrainerWithDAdamW(SFTTrainer):
    def __init__(self, *args, preconditioner_power=0.5, **kwargs):
        # Call parent __init__ with all its arguments so I pass that correctly
        super().__init__(*args, **kwargs)
        # Then I can set custom precond power
        self.preconditioner_power = preconditioner_power

    def create_optimizer(self):
        """Override to use DAdamW instead of default optimizer. A custom trainer is necessary to handle the DS Z-3 stuff correctly"""
        if self.optimizer is None:
            # Model is already wrapped by Accelerator at this point
            optimizer = setup_dadamw(self.args, self.model, self.preconditioner_power)
            self.optimizer = optimizer
        return self.optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        """Override to use custom scheduler"""
        print("Currently warmup_ratio hardcoded to 0.1!")
        if optimizer is None:
            optimizer = self.optimizer
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * num_training_steps)  # warmup_ratio=0.1
        )
        return self.lr_scheduler
    
    
### What follows is the default compute_loss() function of TRL's SFTTrainer
#
#
#
#
###
# need some extra imports from TRL to avoid PyLance from complaining about a bunch of "is not defined"

# from trl.trainer.utils import entropy_from_logits
# from transformers.utils import is_peft_available
# if is_peft_available():
#     from peft import PeftConfig, PeftModel, PeftType, get_peft_model

# def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         mode = "train" if self.model.training else "eval"

#         # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
#         # This can be removed when this issue is fixed.
#         # When using CP or SP, labels are pre-shifted, we must use shift_labels instead.
#         labels = inputs["labels"] if "shift_labels" not in inputs else None

#         # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
#         inputs["use_cache"] = False

#         # Request token accuracy from Liger kernel and set token scaling if using DFT loss
#         if self.args.use_liger_kernel:
#             inputs["return_token_accuracy"] = True
#             inputs["use_token_scaling"] = self.args.loss_type == "dft"

#         (loss, outputs) = super().compute_loss(
#             model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
#         )

#         # Compute entropy
#         if not self.args.use_liger_kernel:  # liger doesn't return logits
#             with torch.no_grad():
#                 per_token_entropy = entropy_from_logits(outputs.logits)
#                 # When using Prompt Tuning, skip the virtual tokens in logits before entropy computation, since they
#                 # do not correspond to actual input tokens.
#                 if (
#                     self.num_virtual_tokens > 0
#                     and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
#                 ):
#                     per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
#                 if "attention_mask" in inputs:
#                     attention_mask = inputs["attention_mask"]
#                     entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
#                 elif "position_ids" in inputs:
#                     entropy = torch.mean(per_token_entropy)
#                 else:
#                     raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
#                 entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
#             self._metrics[mode]["entropy"].append(entropy)

#         if mode == "train":
#             # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
#             # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
#             if "attention_mask" in inputs:
#                 num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
#             elif "position_ids" in inputs:
#                 local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
#                 num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
#             else:
#                 raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
#             self._total_train_tokens += num_tokens_in_batch
#         self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

#         if self.args.use_liger_kernel:
#             token_accuracy = self.accelerator.gather_for_metrics(outputs.token_accuracy).mean().item()
#             self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
#         else:
#             # Compute accuracy from logits using argmax (traditional method)
#             with torch.no_grad():
#                 if "shift_labels" in inputs:
#                     # When using CP or SP, labels are pre-shifted. We must use these (and cannot manually shift) because:
#                     # - The first discarded token from inputs["labels"] actually belongs to process n-1
#                     # - The last logits require the label from process n+1
#                     shift_logits = outputs.logits.contiguous()
#                     shift_labels = inputs["shift_labels"]
#                 else:
#                     shift_logits = outputs.logits[..., :-1, :].contiguous()
#                     shift_labels = labels[..., 1:].contiguous()

#                 # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
#                 if (
#                     self.num_virtual_tokens > 0
#                     and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
#                 ):
#                     shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

#                 # Get predictions
#                 predictions = shift_logits.argmax(dim=-1)

#                 # Create mask for non-padding tokens (assuming ignore_index is -100)
#                 mask = shift_labels != -100

#                 # Calculate accuracy only on non-padding tokens
#                 correct_predictions = (predictions == shift_labels) & mask
#                 total_tokens = mask.sum()
#                 correct_tokens = correct_predictions.sum()

#                 # Gather the correct_tokens and total_tokens across all processes
#                 correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
#                 total_tokens = self.accelerator.gather_for_metrics(total_tokens)

#                 # Compute the mean token accuracy and log it
#                 total_sum = total_tokens.sum()
#                 accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
#                 self._metrics[mode]["mean_token_accuracy"].append(accuracy)

#         # Log auxiliary loss if enabled (applies to both Liger and non-Liger)
#         if self.aux_loss_enabled:
#             aux_loss = outputs.aux_loss
#             aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
#             self._metrics[mode]["aux_loss"].append(aux_loss)

#         return (loss, outputs) if return_outputs else loss
    
    
### Above is the untouched, default compute_loss() function of TRL's SFTTrainer
#
#
#
#
###


# TRL's Trainer accepts a custom `compute_loss_func` when instantiating the trainer: 
# """
# compute_loss_func (Callable, optional) — A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default loss function used by Trainer.
# """"
# in training_step(), trainer calls 
# ``` loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch) ``
# and then later
# ``` self.accelerator.backward(loss, **kwargs) ```

# SFTTrainer inherits this, so all we need write is 
# - code to compute Fisher
# - compute_loss_func that uses fisher in calculating total loss
# this loss will be passed to trainer's training_step(), which will do backward() wrt our Fisher-respecting loss


### update Jan'25: I was going to use compute_loss_func, but the call indeed is 🤗 transformers trainer.py L4123
# loss = self.compute_loss_func(
            #     outputs,
            #     labels,
            #     num_items_in_batch=num_items_in_batch,
            # )
# so we don't get access to the model! Need to subclass after all...

class FisherCallback(TrainerCallback):
    """Compute Fisher Information after model is on device but before training starts."""
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # self.trainer is set by the trainer when adding callback
        trainer = self.trainer
        print("Fisher: calling FisherCallback in hopes model is properly wrapped with accelerator")
        print("Fisher: BSZ is still 1 to avoid OOM!")
        if not trainer.fisher_initialised:
            trainer._compute_fisher_distributed()
    

class SFTTrainerWithFisher(SFTTrainer):
    """
    SFTTrainerWithFisher implements Elastic Weight Consolidation, compatible with FSDP and DeepSpeed ZeRO stages 1-2 (Stage 3 will require a rewrite, FIM wrt. sharded model params is finnicky). 

    Args:
        args ([`TrainingArguments`]):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`].
        retain_dataset_id (`string`):
            The 🤗 identifier of the dataset to use for commputing Fisher Information.
        ewc_lambda (`float`):
            The lambda weight that scales the EWC loss during training, e.g. 50.0.
        fisher_batch_size (`int`):
            The batch size to use while computing Fisher Information.
        recompute_fisher_mode (`string`):
            The strategy to use for recomputing Fisher Information. Must be one of 'never', 'intervals', or 'dynamically'.
        recompute_fisher_intervals (`float`, *optional*):
            The intervals at which Fisher Information should be recomputed, for e.g. 0.25. Must be set if recompute_fisher_mode == 'intervals'.
        fisher_num_batches (`int`):
            The number of batches with respect to compute Fisher Information.
        fisher_completion_only_loss (`bool`):
            If using completion_only_loss=True in SFT, Fisher loss computation should use this too for gradients to be aligned.
            """
    def __init__(
        self, 
        *args, 
        retain_dataset_id: str,
        ewc_lambda: float,
        fisher_batch_size: int,
        recompute_fisher_mode: str,
        recompute_fisher_intervals: float, 
        fisher_num_batches: int,
        fisher_completion_only_loss: bool,
        **kwargs
        ):
        print(f"Fisher: Init with ewc_lambda {ewc_lambda}")
        super().__init__(*args, **kwargs)
        self.retain_dataset_id = retain_dataset_id
        self.ewc_lambda = ewc_lambda
        self.fisher_batch_size = fisher_batch_size
        self.recompute_fisher_mode = recompute_fisher_mode
        self.recompute_fisher_intervals = recompute_fisher_intervals
        self.fisher_num_batches = fisher_num_batches
        self.fisher_completion_only_loss = fisher_completion_only_loss
        with self.accelerator.main_process_first(): #otherwise we preprocess num_gpu times
            raw_dataloader = self.preprocess_retain_dataset(retain_dataset_id)
        
        # super().__init_() calls SFTTrainer init which calls Trainer init, so we should have an accelerator already 
        # so this should prep the dataloader for num_gpus
        self.retain_dataloader = self.accelerator.prepare(raw_dataloader)
        self.fisher_initialised = False

        fisher_callback = FisherCallback()
        fisher_callback.trainer = self  # Give callback access to trainer
        self.add_callback(fisher_callback)

        
    def preprocess_retain_dataset(self, retain_dataset_id: str, ):
        print("Fisher: Pre-processing dataset")
        # if using self.tokenizer, we get thousands of "Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead." because of .map()
        tokenizer = self.processing_class 
        print("Generation tag present:", "{% generation %}" in tokenizer.chat_template)
        print("Endgeneration tag present:", "{% endgeneration %}" in tokenizer.chat_template)

        # If False, print a snippet around where it should be
        if "{% generation %}" not in tokenizer.chat_template:
            # Find assistant handling section
            idx = tokenizer.chat_template.find("assistant")
            print("Around 'assistant':", repr(tokenizer.chat_template[idx:idx+200]))


        # making sure the chat template was passed reasoning format and generation format correctly before filtering with it
        assert "<think>\n...\n</think>\n<answer>\n...\n</answer>\"" in tokenizer.chat_template
        assert "{% generation %}" in tokenizer.chat_template
        print('num proc still hardcoded')
        num_proc = 16
        max_length = self.args.max_length
        raw_dataset = load_dataset(retain_dataset_id)["train"]
        # TODO: properly implement fisher_completion_only_loss toggle
        
        # tokenize to check full lengths of sequences
        def preprocess(example):
            tokenized = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                return_assistant_tokens_mask=self.fisher_completion_only_loss,
                return_dict=True,
            )
            if self.fisher_completion_only_loss: # if we want to support completion_only_loss=True in SFT, we need to align Fisher loss computation.
                return_dict = {
                    "input_ids": tokenized["input_ids"],
                    "assistant_masks": tokenized["assistant_masks"], 
                }
            else:
                return_dict = {
                    "input_ids": tokenized["input_ids"],
                }
            return return_dict
        tokenized_dataset = raw_dataset.map(
            preprocess, 
            remove_columns=raw_dataset.column_names, 
            num_proc=num_proc, 
            batched=True,
            batch_size=1000,
            desc="Tokenizing"
            )
        
        # then filter to max length
        def shorter_than(example):
            return len(example["input_ids"]) <= max_length
        final_dataset = tokenized_dataset.filter(shorter_than, num_proc=num_proc, desc=f"Filtering retain dataset to max length {max_length}")
        print(f"Original retain dataset length: {len(tokenized_dataset)}, retain dataset length after filtering: {len(final_dataset)}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # and prep 
        collator = DataCollatorForLanguageModeling(pad_token_id=self.tokenizer.pad_token_id, completion_only_loss=True)
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.fisher_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        return dataloader
    
    def _compute_fisher_distributed(self,):

        # TODO: implement this with DeepSpeed ZeRO stage 3 using 
        #   with deepspeed.zero.GatheredParameters(param.weight, modifier_rank=0):
        print("Fisher: preparing to (re)compute fisher")
        self.accelerator.wait_for_everyone() # so that DeepSpeed setup finishes printing to CLI before fisher info is printed across all devices

        
        if self.accelerator.is_main_process:
            print("Fisher: computing FIM across all devices")
        print(f"Fisher: rank {self.accelerator.process_index} on {self.accelerator.device}")


        self.model.eval()
        
        self.fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        
        self.reference_params = {
            name: param.detach().clone() 
            for name, param in self.model.named_parameters() 
            if param.requires_grad
        }
        
        num_batches = 0
        batches_per_device = math.ceil(self.fisher_num_batches / self.accelerator.num_processes)

        for batch in tqdm(self.retain_dataloader, desc="Computing Fisher (total shows fisher_num_batches/num_gpus )", disable=not self.accelerator.is_main_process, total=batches_per_device): 
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.detach().pow(2)
            
            num_batches += 1
            if num_batches >= batches_per_device:
                break


        self.accelerator.wait_for_everyone() # critical to sync before all-reduce to ensure all ranks have finished their portion of the retain dataset
        for name in self.fisher:
            self.fisher[name] /= num_batches # this should be a local averaging
        print(f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

        if self.accelerator.num_processes > 1:
            for name in self.fisher:
                dist.all_reduce(self.fisher[name], op=dist.ReduceOp.SUM) # sync between ranks to 'accumulate' fisher info
                self.fisher[name] /= self.accelerator.num_processes # then divide by num ranks to get the average

        self.accelerator.wait_for_everyone()  # another sync to ensure we're ready everywhere before training starts

        self.fisher_initialised = True
        self.model.zero_grad(set_to_none=True) # quite critical to ensure i don't add gradients into model before actual training starts
        torch.cuda.empty_cache()
        self.model.train()
        if self.accelerator.is_main_process:
            print("Fisher: done")
                
        
        return
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Debug: check Fisher types
        if not hasattr(self, '_fisher_debug_done'):
            for name, val in list(self.fisher.items())[:3]:
                print(f"Fisher type check: {name} -> {type(val)}, shape: {getattr(val, 'shape', 'N/A')}")
            self._fisher_debug_done = True
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                ewc_loss += (self.fisher[name] * (param - self.reference_params[name]).pow(2)).sum()
        
        loss = loss + self.ewc_lambda * ewc_loss
        
        return (loss, outputs) if return_outputs else loss