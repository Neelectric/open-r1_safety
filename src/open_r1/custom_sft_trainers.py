"""Implements DAdamW (dampened AdamW, where the preconditioner power can be varied as needed), as well as training with Fisher Information (e.g. for EWC)."""

import math
from typing import Any, Callable, Optional, Union

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
        if wandb.run is not None:
            wandb.run.config.update({
                "recompute_fisher_mode": trainer.recompute_fisher_mode, 
                "recompute_fisher_intervals": trainer.recompute_fisher_intervals,
                "retain_dataset_id": trainer.retain_dataset_id,
                "ewc_lambda": trainer.ewc_lambda,
                "fisher_batch_size": trainer.fisher_batch_size,
                "fisher_num_batches": trainer.fisher_num_batches,
                "fisher_completion_only_loss": trainer.fisher_completion_only_loss,
                })
            print("Fisher: Updated config with all vars")
        else: 
            print("wandb.run is None?")


        
    
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
        
        # simple check all variables are reasonable
        assert (type(ewc_lambda) == float) and (ewc_lambda > 25.0), "probably needs to be higher than 25.0?"
        assert (type(fisher_batch_size) == int) and (fisher_batch_size > 0) 
        assert recompute_fisher_mode == "never", "recomputation modes 'intervals' and 'dynamically' are not yet implemented"
        assert (recompute_fisher_intervals < 1.0)
        assert (fisher_num_batches > 100)
        
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
        # small_subset_size = 2000
        # print(f"Still only using {small_subset_size} samples of retain!!! " * 10)
        # raw_dataset = raw_dataset.select(range(0,small_subset_size))
        
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
            batch_size=10,
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
        print_mem = False
        print_debug = False
        if print_mem:
            print("Fisher: before super compute loss"
                f"Rank {self.accelerator.process_index}: "
                f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        if print_debug:
            # Debug: check Fisher types
            if not hasattr(self, '_fisher_debug_done'):
                for name, val in list(self.fisher.items())[:3]:
                    print(f"Fisher type check: {name} -> {type(val)}, shape: {getattr(val, 'shape', 'N/A')}, dtype: {getattr(val, 'dtype', 'N/A')}, requires_grad: {getattr(val, 'requires_grad', 'N/A')}")
                for name, val in list(self.reference_params.items())[:3]:
                    print(f"Ref params type check: {name} -> {type(val)}, shape: {getattr(val, 'shape', 'N/A')}, dtype: {getattr(val, 'dtype', 'N/A')}, requires_grad: {getattr(val, 'requires_grad', 'N/A')}")
                    
                self._fisher_debug_done = True
                
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        
        if print_mem:
            print("Fisher: after super compute loss"
                f"Rank {self.accelerator.process_index}: "
                f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        
        # assert self.accelerator.num_processes == 1, "don't think multi-GPU will train correctly yet - need to look into all-reduce of ewc_loss, though i think it should be the same on all devices?"
        # update: in Z-1, each device will have a different batch, and hence different loss and different local gradients
        # to ensure that everything is correct when synchronization happens before optimizer causes param update, 
        # i suspect it is important for us to all-gather ewc_loss
        
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if "module." in name:
                name = name.replace("module.", "")
            if name in self.fisher:
                # L(θ) = LB (θ) +  ∑_i  λ/2 F_i (θ_i − θ_{A,i}^*)^2
                diff_of_params = (param - self.reference_params[name])
                sq_diff_of_params = diff_of_params.pow(2)
                fisher_times_sq_diff_of_params = (self.fisher[name] * sq_diff_of_params)
                sum_fisher_times_sq_diff_of_params = fisher_times_sq_diff_of_params.sum()
                full = (self.fisher[name] * (param - self.reference_params[name]).pow(2)).sum()
                ewc_loss += (self.fisher[name] * (param - self.reference_params[name]).pow(2)).sum()
            else:
                print(f"Fisher: Encountered param {name} which is not fisher. Continuing training as param.requires_grad may have intentionally been False at Fisher instantiation")
                
                
        # edit: I was unusure whether we would need to all_reduce ewc_loss across accelerators before adding it to our loss. However, when using DeepSpeed ZeRO Stage 1 this prints a lot of
        # "Fisher: Rank 1 - ewc_loss before all_reduce SUM and avg 1.4823076576950256e-19"
        # "Fisher: Rank 1 - ewc_loss after all_reduce SUM and avg 1.4823076576950256e-19"
        # So in Stage 1 im fairly certain we don't need this
        
        assert model.optimizer.zero_stage_string == "ZeRO-1", "I need to revisit 'ZeRO-2' to make check whether we need to all_reduce ewc_loss on every compute_loss call..."
        
        if model.optimizer.zero_stage_string != "ZeRO-1":
            # according to https://docs.pytorch.org/docs/stable/distributed.html,
            # AVG divides values by the world size before summing across ranks. AVG is only available with the NCCL backend, and only for NCCL versions 2.10 or later.
            # so we will do it manually for now
            
            print(f"Fisher: Rank {self.accelerator.process_index} - ewc_loss before all_reduce SUM and avg {ewc_loss}") # this prints stuff like: 
            dist.all_reduce(ewc_loss, op=dist.ReduceOp.SUM) #
            ewc_loss = ewc_loss / self.accelerator.num_processes
            print(f"Fisher: Rank {self.accelerator.process_index} - ewc_loss after all_reduce SUM and avg {ewc_loss}") # this prints stuff like: 
            
            # also this currently seems to cause:
            
            # /root/openr1_v2/lib/python3.12/site-packages/torch/autograd/graph.py:841: UserWarning: c10d::allreduce_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at /pytorch/torch/csrc/autograd/autograd_not_implemented_fallback.cpp:62.)
            
            # which is concerning. need to read up on how all_reduce interacts with Autograd/backprop
        
        
        # apply lambda, add to loss, log and return
        loss = loss + self.ewc_lambda * ewc_loss
        if print_mem:
            print("Fisher: after ewc compute loss"
                f"Rank {self.accelerator.process_index}: "
                f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
            
        # AFAICT we can't trivially access self.log() here so storing it as an attr for CallBack to log with on_log()
        self._ewc_loss = ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else float(ewc_loss)
        if print_debug:
            print(f"Fisher: ewc_loss: {ewc_loss}")
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Overrides Trainer.log() to inject custom behavior of tracking ewc_loss

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if hasattr(self, '_ewc_loss'):
            logs['ewc_loss'] = self._ewc_loss
            if wandb.run is not None:
                    wandb.log({"train/ewc_loss": self._ewc_loss}, step=self.state.global_step)
            
        super().log(logs, start_time)