uv pip install lighteval==0.9.2
uv pip install vllm==0.8.5
uv pip install more_itertools
uv pip install datasets==2.21.0

NUM_GPUS=1
NUM_TOKS=4096

# MODEL=/root/.cache/huggingface/hub/models--Neelectric--Llama-3.1-8B-Instruct_GRPO_Math-220kv00.10/snapshots/afbdbaa07751effee4a6f40b2d5b77ea1a876435
# REVISION=main-step-000000275
# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.10
MODEL=meta-llama/Llama-3.1-8B-Instruct
REVISION=main


# ifbench_test
MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/




TASK=ifbench_test
VLLM_WORKER_MULTIPROC_METHOD="spawn" \
CUDA_VISIBLE_DEVICES="0" \
TORCHINDUCTOR_CACHE_DIR=./.cache/${CUDA_VISIBLE_DEVICES}/ \
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

TASK=ifeval
VLLM_WORKER_MULTIPROC_METHOD="spawn" \
CUDA_VISIBLE_DEVICES="0" \
TORCHINDUCTOR_CACHE_DIR=./.cache/${CUDA_VISIBLE_DEVICES}/ \
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR









uv pip install vllm==0.11.2
uv pip install datasets==4.4.2
