# uv pip install lighteval==0.13.0
uv pip install vllm==0.10.1
uv pip install more_itertools syllapy "spacy[ja,ko,th]>=3.8.0" emoji "numpy==2.2"

NUM_GPUS=4
NUM_TOKS=4096

# MODEL=/root/.cache/huggingface/hub/models--Neelectric--Llama-3.1-8B-Instruct_GRPO_Math-220kv00.10/snapshots/afbdbaa07751effee4a6f40b2d5b77ea1a876435
# REVISION=main-step-000000275
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.21
REVISION=main


# ifeval
MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD="spawn" \
lighteval vllm $MODEL_ARGS ifeval,ifbench_test \
    --output-dir $OUTPUT_DIR


uv pip install vllm==0.11.2