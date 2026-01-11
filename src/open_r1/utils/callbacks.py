#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import torch
import wandb

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config, **kwargs) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            # if is_slurm_available():
            #     dummy_config.benchmarks = args.benchmarks

            #     def run_benchmark_callback(_):
            #         print(f"Checkpoint {global_step} pushed to hub.")
            #         run_benchmark_jobs(dummy_config, self.model_config)

            #     future.add_done_callback(run_benchmark_callback)


class BenchmarkCallback(TrainerCallback):
    """
    Callback that runs lighteval benchmarks on the model at each checkpoint save.

    Benchmarks to run should be specified in the training config's `benchmarks` field.
    Results are saved to ./results/checkpoint-{step}/ directory.
    """

    def __init__(self, model_config, **kwargs) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only run on main process to avoid duplicate evaluations
        if not state.is_world_process_zero:
            return

        # Skip if no benchmarks specified
        if not hasattr(args, "benchmarks") or not args.benchmarks:
            print("No benchmarks specified in config, skipping benchmark evaluation.")
            return

        try:
            from lighteval.logging.evaluation_tracker import EvaluationTracker
            from lighteval.models.transformers.transformers_model import (
                TransformersModel,
                TransformersModelConfig,
            )
            from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
        except ImportError as e:
            print(f"Warning: lighteval not installed, skipping benchmark evaluation. Error: {e}")
            return

        global_step = state.global_step
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")

        if model is None:
            print("Warning: model not found in callback kwargs, skipping benchmark evaluation.")
            return

        # Create output directory for this checkpoint's results
        results_dir = os.path.join(args.output_dir, f"lighteval_results/checkpoint-{global_step}")
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Running lighteval benchmarks at step {global_step}")
        print(f"Benchmarks: {', '.join(args.benchmarks)}")
        print(f"Results will be saved to: {results_dir}")
        print(f"{'='*80}\n")

        # Set up evaluation tracker
        evaluation_tracker = EvaluationTracker(
            output_dir=results_dir,
            save_details=True,
            push_to_hub=True,  
            hub_results_org="Neelectric"
        )

        # Set up pipeline parameters
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            override_batch_size=1,  # Use small batch size to avoid OOM during training
        )

        # Get the base model (unwrap from PEFT if needed)
        base_model = model
        if hasattr(model, "get_base_model"):
            base_model = model.get_base_model()

        # Configure the model for lighteval
        # Use the model's name from config if available
        model_name = getattr(self.model_config, "model_name_or_path", "model")

        lighteval_config = TransformersModelConfig(
            model_name=model_name,
            batch_size=1,
        )

        # Wrap the model with TransformersModel
        lighteval_model = TransformersModel.from_model(
            model=base_model,
            config=lighteval_config,
            tokenizer=tokenizer,
        )

        # Create and run pipeline
        tasks = ",".join(args.benchmarks)  # lighteval accepts comma-separated task names

        pipeline = Pipeline(
            model=lighteval_model,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            tasks=tasks,
        )

        try:
            print(f"Starting evaluation...")
            results = pipeline.evaluate()
            pipeline.save_and_push_results()
            pipeline.show_results()

            print(f"\n{'='*80}")
            print(f"Benchmark evaluation completed at step {global_step}")
            print(f"Results saved to: {results_dir}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"Error during benchmark evaluation: {e}")
            import traceback
            traceback.print_exc()

class EMACallback(TrainerCallback):
    def __init__(self, train_config=None, model_config=None, **kwargs):
        self.eta = getattr(train_config, 'ema_eta', 0.25) if train_config else 0.25
        self.ema_params = []
        self.initialized = False
        
    def on_train_begin(self, args, state, control, model, **kwargs):
        if not self.initialized:
            # this should handle DDP wrapping
            self.model_ref = model.module if hasattr(model, "module") else model
            self.ema_params = [
                p.clone().detach() 
                for p in self.model_ref.parameters() 
                if p.requires_grad
            ]
            self.initialized = True
            print(f"Simple EMA: Copied Model params into ema_params with eta = {self.eta}")

    def on_step_end(self, args, state, control, model, **kwargs):
        # apply actual EMA: ema = eta * ema + (1-eta) * current
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_params, self.model_ref.parameters()):
                if model_p.requires_grad:
                    ema_p.mul_(self.eta).add_(model_p, alpha=(1 - self.eta))
                    
    def on_log(self, args, state, control, logs, **kwargs):
        if state.global_step % 50 == 0 and state.global_step > 0:
            with torch.no_grad():
                param_dist = sum(
                    (e - m).pow(2).sum() 
                    for e, m in zip(self.ema_params, self.model_ref.parameters())
                    if m.requires_grad
                ).sqrt().item()
            print(f"EMA: Param_dist is {param_dist}")
            logs["ema_param_distance"] = param_dist
            if "wandb" in args.report_to:
                if wandb.run is not None:
                    wandb.log({"ema_param_distance": param_dist}, step=state.global_step)

                    
    def on_train_end(self, args, state, control, model, **kwargs):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_params, self.model_ref.parameters()):
                if model_p.requires_grad:
                    model_p.copy_(ema_p)
        print(f"Successfully replaced all model params with the EMA params!")
                
class ShardedEMACallback(TrainerCallback):
    """
    EMA callback that is intended to work with DeepSpeed ZeRO stages 1 through 3.
    Operates on optimizer param shards directly which should help avoid communication overhead and minimizes per-GPU VRAM overhead
    """
    def __init__(self, train_config=None, model_config=None, script_args=None, **kwargs):
        self.eta = getattr(script_args, 'ema_eta', 0.25) if script_args else 0.25
        print(f"self.eta is {self.eta}")
        self.ema_param_groups = []  # Fixed: was ema_params but referenced as ema_param_groups
        self.initialized = False
        self.base_optimizer = None
        
    def _get_base_optimizer(self, optimizer):
        """Unwrap DeepSpeed optimizer to get the underlying optimizer."""
        if hasattr(optimizer, 'optimizer'):
            return optimizer.optimizer
        return optimizer
        
    def on_train_begin(self, args, state, control, optimizer, **kwargs):
        if self.initialized:
            return
            
        self.base_optimizer = self._get_base_optimizer(optimizer)
        
        # Create shadow copy of local optimizer shards
        # This automatically respects ZeRO partitioning
        for group in self.base_optimizer.param_groups:
            self.ema_param_groups.append([
                p.clone().detach() 
                for p in group['params'] 
                if p.requires_grad
            ])
        
        self.initialized = True
        
        # Log some diagnostics
        total_params = sum(
            p.numel() for group in self.base_optimizer.param_groups 
            for p in group['params'] if p.requires_grad
        )
        if args.local_rank <= 0:
            print(f"ShardedEMA initialized: eta={self.eta}, local shard params={total_params:,}")

    def on_step_end(self, args, state, control, optimizer, **kwargs):
        if not self.initialized:
            # fallback init just in case if on_train_begin didn't receive optimizer
            self.on_train_begin(args, state, control, optimizer, **kwargs)
        
        base_opt = self._get_base_optimizer(optimizer)
        
        # apply EMA only to local shards
        # as far as I can tell this should be mathematically equivalent to global EMA since shards are disjoint
        with torch.no_grad():
            for ema_group, opt_group in zip(self.ema_param_groups, base_opt.param_groups):
                for ema_p, opt_p in zip(ema_group, opt_group['params']):
                    if opt_p.requires_grad:
                        ema_p.mul_(self.eta).add_(opt_p, alpha=(1 - self.eta))
    
    def on_log(self, args, state, control, logs, optimizer, **kwargs):
        if state.global_step % 50 != 0 or state.global_step == 0:
            return
            
        if not self.initialized:
            return
            
        base_opt = self._get_base_optimizer(optimizer)
        
        # Compute local shard distance
        with torch.no_grad():
            local_dist_sq = sum(
                (ema_p - opt_p).pow(2).sum()
                for ema_group, opt_group in zip(self.ema_param_groups, base_opt.param_groups)
                for ema_p, opt_p in zip(ema_group, opt_group['params'])
                if opt_p.requires_grad
            )
            
            # All-reduce to get global distance across shards
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_dist_sq, op=torch.distributed.ReduceOp.SUM)
            
            param_dist = local_dist_sq.sqrt().item()
        
        if args.local_rank <= 0:
            print(f"ShardedEMA step {state.global_step}: param_dist={param_dist:.4f}")
            if "wandb" in args.report_to:
                if wandb.run is not None:
                    wandb.log({"ema_param_distance": param_dist}, step=state.global_step)
                
        logs["ema_param_distance"] = param_dist
        
    
    def on_save(self, args, state, control, **kwargs):
        """Save EMA state alongside checkpoints."""
        if args.local_rank <= 0:
            import os
            ema_path = os.path.join(args.output_dir, f"ema_state_{state.global_step}.pt")
            # Only save on rank 0 - each rank has different shards
            # For full recovery, would need to save per-rank, but this is mainly for debugging
            torch.save({
                'eta': self.eta,
                'global_step': state.global_step,
                'ema_param_shapes': [[p.shape for p in group] for group in self.ema_param_groups],
            }, ema_path)
            print(f"ShardedEMA: Saved metadata to {ema_path}")
    
    def on_train_end(self, args, state, control, optimizer, **kwargs):
        if not self.initialized:
            print("ShardedEMA: Warning - not initialized, skipping param replacement")
            return
            
        base_opt = self._get_base_optimizer(optimizer)
        
        # Copy EMA params back to optimizer's param references
        with torch.no_grad():
            for ema_group, opt_group in zip(self.ema_param_groups, base_opt.param_groups):
                for ema_p, opt_p in zip(ema_group, opt_group['params']):
                    if opt_p.requires_grad:
                        opt_p.copy_(ema_p)
        
        if args.local_rank <= 0:
            print(f"ShardedEMA: Replaced optimizer params with EMA params")
        
        # Synchronize to ensure all ranks complete before any saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "benchmark_callback": BenchmarkCallback,
    "ema_callback": EMACallback,
    "sharded_ema_callback": ShardedEMACallback,
}


def get_callbacks(train_config, model_config, script_args=None) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](
            train_config=train_config, 
            model_config=model_config,
            script_args=script_args,
        ))
    return callbacks
