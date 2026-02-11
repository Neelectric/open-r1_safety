MODEL_ID="${1:-Neelectric/Llama-3.1-8B-Instruct_SFT_codeforcesv00.01}"
REVISION="${2:-main}"
NUM_GPUS="${3:-4}"

uv pip install vllm==0.10.1
uv pip uninstall flashinfer-python
uv pip install more_itertools

# NUM_GPUS=4
# 4096
NUM_TOKS=16384
echo $NUM_TOKS
# NUM_TOKS=32768
GPU_MEM_UTIL=0.95


# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.30
# REVISION=main

# 324
MODEL_ARGS="model_name=$MODEL_ID,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=$GPU_MEM_UTIL,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
TASK=lcb:codegeneration
# lcb:codegeneration_v2
# lcb:codegeneration_release_v1
# lcb:codegeneration_v1_v4
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD="spawn" \
lighteval vllm $MODEL_ARGS $TASK \
    --output-dir $OUTPUT_DIR

uv pip install vllm==0.11.2
uv pip install flashinfer-python

