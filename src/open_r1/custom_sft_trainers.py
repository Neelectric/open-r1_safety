"""Implements DAdamW (dampened AdamW, where the preconditioner power can be varied as needed), as well as training with Fisher Information (e.g. for EWC)."""

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

import wandb

from tqdm import tqdm

from open_r1.optims.dadamw import DAdamW, setup_dadamw


class SFTTrainerWithDAdamW(SFTTrainer):
    def __init__(self, *args, preconditioner_power=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.preconditioner_power = preconditioner_power

    def create_optimizer(self):
        """Override to use DAdamW instead of default optimizer. A custom trainer is necessary to handle DeepSpeed ZeRO-3 correctly"""
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


# Originally implemented fisher computation inside an overriden _inner_training_loop(), but only at the start of that is the model actually wrapped appropriately - so we need a callback.
class FisherCallback(TrainerCallback):
    """Compute Fisher Information after model is on device but before training starts."""
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainer = self.trainer
        print("Fisher: calling FisherCallback in hopes model is properly wrapped with accelerator")
        print("Fisher: BSZ is still 1 to avoid OOM!")
        if not trainer.fisher_initialised:
            trainer._compute_fisher_distributed()
    
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

### update: I was going to use compute_loss_func, but the call indeed is 🤗 transformers trainer.py L4123
# loss = self.compute_loss_func(
            #     outputs,
            #     labels,
            #     num_items_in_batch=num_items_in_batch,
            # )
# so we don't get access to the model! Need to subclass the trainer after all...

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
        processing_class (`PreTrainedTokenizerBase`):
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
        processing_class,
        **kwargs
        ):
        print(f"Fisher: Init with ewc_lambda {ewc_lambda}")
        self.retain_dataset_id = retain_dataset_id
        self.ewc_lambda = ewc_lambda
        self.fisher_batch_size = fisher_batch_size
        self.recompute_fisher_mode = recompute_fisher_mode
        self.recompute_fisher_intervals = recompute_fisher_intervals
        self.fisher_num_batches = fisher_num_batches
        self.fisher_completion_only_loss = fisher_completion_only_loss
        
        # we need to preprocess dataset first, otherwise pod gets OOMKilled
        # it's possible multiprocessing in .map() forks the process, and 
        # each fork copies entire memory space including the model?
        self.raw_retain_dataset = self.preprocess_retain_dataset(
            retain_dataset_id=retain_dataset_id,
            processing_class=processing_class,
            max_length=kwargs.get('args').max_length,
            fisher_completion_only_loss=fisher_completion_only_loss,
        )
        
        super().__init__(*args, processing_class=processing_class, **kwargs)
        
        with self.accelerator.main_process_first(): #otherwise we preprocess num_gpu times
            collator = DataCollatorForLanguageModeling(
                pad_token_id=processing_class.pad_token_id, 
                completion_only_loss=self.fisher_completion_only_loss
            )
            raw_dataloader = DataLoader(
                self.raw_retain_dataset,
                batch_size=self.fisher_batch_size,
                shuffle=True,
                collate_fn=collator,
            )
        
        # super().__init_() calls SFTTrainer init which calls Trainer init, so we should have an accelerator already 
        # so this should prep the dataloader for num_gpus
        self.retain_dataloader = self.accelerator.prepare(raw_dataloader)
        self.fisher_initialised = False

        fisher_callback = FisherCallback()
        fisher_callback.trainer = self  # Give callback access to trainer
        self.add_callback(fisher_callback)

    @staticmethod # needs to be static because it gets called before super().__init__(), but we need access to its attrs
    def preprocess_retain_dataset(retain_dataset_id: str, processing_class, max_length, fisher_completion_only_loss):
        print(f"Fisher: Pre-processing dataset while respecting self.fisher_completion_only_loss = {fisher_completion_only_loss}")
        # if using self.tokenizer, we get thousands of "Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead." because of .map()
        tokenizer = processing_class 

        # making sure the chat template was passed reasoning format and generation format correctly before filtering with it
        assert "<think>\n...\n</think>\n<answer>\n...\n</answer>\"" in tokenizer.chat_template
        assert "{% generation %}" in tokenizer.chat_template
        print('num proc still hardcoded')
        num_proc = 1
        raw_dataset = load_dataset(retain_dataset_id)["train"]
        small_subset_size = 2000
        print(f"Still only using {small_subset_size} samples of retain!!! " * 10)
        raw_dataset = raw_dataset.select(range(0,small_subset_size))
        
        # tokenize to check full lengths of sequences
        def preprocess(examples):
            include_mask = fisher_completion_only_loss
            processed = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_assistant_tokens_mask=include_mask,
                    return_dict=True,
                )
                for messages in examples["messages"]
            ]
            
            if fisher_completion_only_loss:
                return {
                    "input_ids": [p["input_ids"] for p in processed],
                    "assistant_masks": [p["assistant_masks"] for p in processed] if include_mask else []
                }
            return {
                "input_ids": [p["input_ids"] for p in processed],
            }

        tokenized_dataset = raw_dataset.map(
            preprocess, 
            remove_columns=raw_dataset.column_names, 
            num_proc=num_proc, 
            batched=True,
            batch_size=50,
            desc=f"Tokenizing retain dataset (num_proc={num_proc})",
            load_from_cache_file=True,
            )
        
        # then filter to max length
        def shorter_than(example):
            return len(example["input_ids"]) <= max_length
        final_dataset = tokenized_dataset.filter(shorter_than, num_proc=num_proc, desc=f"Filtering retain dataset to max length {max_length}")
        print(f"Original retain dataset length: {len(tokenized_dataset)}, retain dataset length after filtering: {len(final_dataset)}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return final_dataset
    
    def _compute_fisher_distributed(self,):
        print("Fisher: before start of fisher computation"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

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
        print("Fisher: we have ",
              f"len(self.retain_dataloader) {len(self.retain_dataloader)}",
              f"self.fisher_num_batches {self.fisher_num_batches}",
              f"self.accelerator.num_processes {self.accelerator.num_processes}",
              f"batches_per_device {batches_per_device}")

        for batch in tqdm(self.retain_dataloader, desc="Computing Fisher (total = fisher_num_batches/num_gpus)", disable=not self.accelerator.is_main_process, total=batches_per_device): 
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
            
        print("Fisher: fisher computation before all reduce"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

        if self.accelerator.num_processes > 1:
            for name in self.fisher:
                dist.all_reduce(self.fisher[name], op=dist.ReduceOp.SUM) # sync between ranks to 'accumulate' fisher info
                self.fisher[name] /= self.accelerator.num_processes # then divide by num ranks to get the average

        self.accelerator.wait_for_everyone()  # another sync to ensure we're ready everywhere before training starts

        self.fisher_initialised = True
        self.model.zero_grad(set_to_none=True) # quite critical to ensure i don't add gradients into model before actual training starts
        del batch, outputs, loss
        torch.cuda.empty_cache()
        self.model.train()
        if self.accelerator.is_main_process:
            print("Fisher: done")
            
        print("Fisher: end of fisher computation"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        return
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        print("Fisher: before super compute loss"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        # Debug: check Fisher types
        if not hasattr(self, '_fisher_debug_done'):
            for name, val in list(self.fisher.items())[:3]:
                print(f"Fisher type check: {name} -> {type(val)}, shape: {getattr(val, 'shape', 'N/A')}, dtype: {getattr(val, 'dtype', 'N/A')}, requires_grad: {getattr(val, 'requires_grad', 'N/A')}")
            for name, val in list(self.reference_params.items())[:3]:
                print(f"Ref params type check: {name} -> {type(val)}, shape: {getattr(val, 'shape', 'N/A')}, dtype: {getattr(val, 'dtype', 'N/A')}, requires_grad: {getattr(val, 'requires_grad', 'N/A')}")
                
            self._fisher_debug_done = True
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        
        print("Fisher: after super compute loss"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                ewc_loss += (self.fisher[name] * (param - self.reference_params[name]).pow(2)).sum()
        
        loss = loss + self.ewc_lambda * ewc_loss
        
        print("Fisher: after ewc compute loss"
            f"Rank {self.accelerator.process_index}: "
            f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
            f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        if wandb.run is not None:
            wandb.log({"ewc_loss": ewc_loss}, step=self.state.global_step)
        
        return (loss, outputs) if return_outputs else loss