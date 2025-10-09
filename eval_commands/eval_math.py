#!/usr/bin/env python3
import os
import multiprocessing

# CRITICAL: Set this BEFORE any imports that use CUDA
multiprocessing.set_start_method('spawn', force=True)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.model_input import GenerationParameters

def main():
    # Configuration
    MODEL_NAME = "Neelectric/Llama-3.1-8B-Instruct_SFT_MoT_mathv00.04"
    OUTPUT_DIR = "data/evals/"

    # Set up evaluation tracking
    evaluation_tracker = EvaluationTracker(
        output_dir=OUTPUT_DIR,
        save_details=True,
        push_to_hub=False,
    )

    # Configure pipeline
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        use_chat_template=True,
    )

    # Configure model
    model_config = VLLMModelConfig(
        model_name=MODEL_NAME,
        dtype="bfloat16",
        data_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_length=32768,
        generation_parameters=GenerationParameters(
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
        ),
    )

    # CRITICAL: Override num_samples to 1
    metric_options = {
        "maj_at_4_math": {"num_samples": 1},
        "pass_at_k_math": {"num_samples": 1},
    }

    # Create and run pipeline
    pipeline = Pipeline(
        tasks="lighteval|math_500|0|0",
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
        metric_options=metric_options,
    )

    results = pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()

if __name__ == '__main__':
    main()