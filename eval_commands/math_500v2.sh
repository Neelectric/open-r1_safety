#!/bin/bash
uv pip install more_itertools

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NUM_GPUS=2
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.07
REVISION=main
MAX_TOKS=4096

VLLM_WORKER_MULTIPROC_METHOD="spawn" \
CUDA_VISIBLE_DEVICES="1" \
TORCHINDUCTOR_CACHE_DIR=./.cache/${CUDA_VISIBLE_DEVICES}/ \
lighteval vllm \
  "model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$MAX_TOKS,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:$MAX_TOKS,temperature:0.6,top_p:0.95}" \
  "custom|math_500_n1|0|0" \
  --custom-tasks math_500_single \
  --use-chat-template \
  --output-dir data/evals/