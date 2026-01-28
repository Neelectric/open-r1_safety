MODEL_ID="${1:-Neelectric/Llama-3.2-1B-Instruct_SFT_Math-220kvdebug_00.04}"
REVISION="${2:-main}"

uv pip install vllm==0.10.1
uv pip install flashinfer-python
uv pip install more_itertools

NUM_GPUS=4
MAX_NEW_TOKENS=4096
MAX_MODEL_LENGTH=8192
# MAX_MODEL_LENGTH=16384
GPU_MEM_UTIL=0.95

# MODEL=/root/.cache/huggingface/hub/models--Neelectric--Llama-3.1-8B-Instruct_GRPO_Math-220kv00.10/snapshots/e2583dfaba3d8acc831710720ff608b0fac3c4a9
# REVISION=main-step-000000025

# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.30
# REVISION=main

MODEL_ARGS="model_name=$MODEL_ID,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$MAX_MODEL_LENGTH,gpu_memory_utilization=$GPU_MEM_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:0.6,top_p:0.95}"
TASK=gpqa:main:0,gpqa:diamond:0
OUTPUT_DIR=data/evals/

# VLLM_USE_RAY_SPMD_WORKER=0 \
VLLM_WORKER_MULTIPROC_METHOD="spawn" \
lighteval vllm $MODEL_ARGS $TASK \
    --output-dir $OUTPUT_DIR



uv pip install vllm==0.11.2
uv pip install flashinfer-python