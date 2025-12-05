



1. **Serve SDG Model:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
   --port 8001 \
   --max-model-len 4096
   ```

2. **Serve Guard Model:**
   ```bash
   CUDA_VISIBLE_DEVICES=1 vllm serve allenai/wildguard \
   --port 8002 \
   --max-model-len 4096
   ```