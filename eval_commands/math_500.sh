uv pip install vllm==0.8.5
uv pip install more_itertools
NUM_GPUS=8
# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_MoT_mathv00.04
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_codeforcesv00.01
REVISION=main
NUM_TOKS=4096

# base
MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$NUM_TOKS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$NUM_TOKS,temperature:0.6,top_p:0.95}"
TASK=math_500
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD="spawn" \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
TORCHINDUCTOR_CACHE_DIR=./.cache/${CUDA_VISIBLE_DEVICES}/ \
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

uv pip install vllm==0.10.2

# #1k
# REVISION=v00.01-step-000001172
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# #2k
# REVISION=v00.01-step-000002344
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# #3k
# REVISION=v00.01-step-000003516
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR


# #4k
# REVISION=v00.01-step-000004688
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR


# #5859k
# REVISION=v00.01-step-000005859
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR
