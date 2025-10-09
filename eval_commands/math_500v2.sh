export VLLM_WORKER_MULTIPROC_METHOD=spawn

lighteval vllm \
  "model_name=Neelectric/Llama-3.1-8B-Instruct_SFT_MoT_mathv00.04,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" \
  "custom|math_500_n1|0|0" \
  --custom-tasks math_500_single \
  --use-chat-template \
  --output-dir data/evals/