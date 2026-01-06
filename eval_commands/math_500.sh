# uv pip install lighteval==0.13.0
uv pip install vllm==0.10.1
# uv pip install -U datasets
uv pip install more_itertools

NUM_GPUS=2
NUM_TOKS=4096

# MODEL=/root/.cache/huggingface/hub/models--Neelectric--Llama-3.1-8B-Instruct_GRPO_Math-220kv00.10/snapshots/e2583dfaba3d8acc831710720ff608b0fac3c4a9
# REVISION=main-step-000000025

MODEL=Neelectric/Llama-3.1-8B-Instruct_GRPO_Math-220kv00.14
REVISION=main-step-000000250


# 324
MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
TASK=math_500
# TASK=aime24
OUTPUT_DIR=data/evals/

# gpuselect --name A6000 -- \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_WORKER_MULTIPROC_METHOD="spawn" \
lighteval vllm $MODEL_ARGS $TASK \
    --output-dir $OUTPUT_DIR



# uv pip install vllm==0.11.2
# uv pip install datasets==4.4.2
