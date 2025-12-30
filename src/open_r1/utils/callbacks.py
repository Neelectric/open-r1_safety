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
    def __init__(self, model_config) -> None:
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

    def __init__(self, model_config) -> None:
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


CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "benchmark_callback": BenchmarkCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
