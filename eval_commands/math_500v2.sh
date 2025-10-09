NUM_GPUS=1
TASK=math_500
OUTPUT_DIR=data/evals/

VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm \
    eval_commands/vllm_config.yaml \
    "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR