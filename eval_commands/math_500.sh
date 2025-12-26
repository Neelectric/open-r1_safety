uv pip install lighteval==0.9.2
uv pip install vllm==0.8.5
uv pip install more_itertools
uv pip install datasets==2.21.0

NUM_GPUS=2
NUM_TOKS=4096

MODEL=/home/user/.cache/huggingface/hub/models--Neelectric--Llama-3.1-8B-Instruct_GRPO_Math-220kv00.06/snapshots/86d8dfa90282309eba1e533431b98737b24440c3
REVISION=v00.06-step-000000405


# 324
MODEL_ARGS="model_name=$MODEL,revision=\"$REVISION\",dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
TASK=math_500
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD="spawn" \
CUDA_VISIBLE_DEVICES="0,1" \
TORCHINDUCTOR_CACHE_DIR=./.cache/${CUDA_VISIBLE_DEVICES}/ \
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR



uv pip install vllm==0.11.0
uv pip install datasets==4.4.2
