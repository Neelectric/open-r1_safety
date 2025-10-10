#!/bin/bash

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=2

lighteval vllm \
  "model_name=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.92,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
  "custom|math_500_n1|0|0" \
  --custom-tasks math_500_single \
  --use-chat-template \
  --output-dir data/evals/