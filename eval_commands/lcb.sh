# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

NUM_GPUS=1
MODEL=meta-llama/Llama-3.1-8B-Instruct

MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=math_500
OUTPUT_DIR=data/evals/$REVISION

VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

# #1k
# REVISION=v00.01-step-000001172
# MODEL_ARGS="model_name=$MODEL,revision=$REVISION,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=math_500
# OUTPUT_DIR=data/evals/$MODEL/$REVISION

# VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --output-dir $OUTPUT_DIR