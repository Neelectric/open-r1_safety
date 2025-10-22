#!/bin/bash

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=1

lighteval vllm \
  "model_name=Neelectric/Llama-3.1-8B-Instruct_GRPO_Math-220k_v00.11,revision=checkpoint-207,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
  "custom|math_500_n1|0|0" \
  --custom-tasks math_500_single \
  --use-chat-template \
  --output-dir data/evals/