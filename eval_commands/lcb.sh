# LiveCodeBench
NUM_GPUS=1
# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_MoT_codev00.03
MODEL=meta-llama/Llama-3.1-8B-Instruct

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=lcb:codegeneration
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 


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