



1. **Serve SDG Model:**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001 --max-model-len 6000 --data-parallel-size=5
   ```

2. **Serve Guard Model:**
   ```bash
   CUDA_VISIBLE_DEVICES=5,6,7 vllm serve allenai/wildguard --port 8002 --max-model-len 7000 --data-parallel-size=3
   ```

uv pip install vllm==0.8.5
uv pip install transformers==4.43

uv pip install vllm==0.11
uv pip install transformers==4.57.3

nohup python sdg.py > output.txt