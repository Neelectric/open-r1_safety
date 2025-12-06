# uv pip install more_itertools
uv pip install vllm==0.8.5
# uv pip install lighteval==0.9.2
uv pip install datasets==2.21.0
# LiveCodeBench
NUM_GPUS=8
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_codeforcesv00.01
# MODEL=meta-llama/Llama-3.1-8B-Instruct
MAX_NEW_TOKENS=16384

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=$MAX_NEW_TOKENS,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:0.6,top_p:0.95}"
TASK=lcb:codegeneration
OUTPUT_DIR=data/evals/

CUDA_VISIBLE_DEIVES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

uv pip install vllm==0.11.0

# #2k
# REVISION=v00.03-step-000002077
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=lcb:codegeneration
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 


# #4k
# REVISION=v00.03-step-000004154
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=lcb:codegeneration
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 


# #6k
# REVISION=v00.03-step-000006231
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=lcb:codegeneration
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 


# #8k
# REVISION=v00.03-step-000008308
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=lcb:codegeneration
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

# #10k
# REVISION=v00.03-step-000010384
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=lcb:codegeneration
# OUTPUT_DIR=data/evals/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 