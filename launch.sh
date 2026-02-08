#!/bin/bash
source /root/openr1_v2/bin/activate

# Configuration
CHECK_INTERVAL=300  # seconds between checks
MEMORY_THRESHOLD=500  # MB - GPUs with less memory used are considered free
UTIL_THRESHOLD=5  # % - GPUs with less utilization are considered free

echo "Waiting for all GPUs to be free..."
echo "Checking every ${CHECK_INTERVAL}s (memory < ${MEMORY_THRESHOLD}MB, util < ${UTIL_THRESHOLD}%)"

while true; do
    # Get memory usage and utilization for all GPUs
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    
    all_free=true
    
    while IFS=, read -r idx mem_used util; do
        mem_used=$(echo "$mem_used" | xargs)
        util=$(echo "$util" | xargs)
        
        if (( mem_used > MEMORY_THRESHOLD )) || (( util > UTIL_THRESHOLD )); then
            all_free=false
            echo "$(date '+%H:%M:%S') - GPU $idx busy (mem: ${mem_used}MB, util: ${util}%)"
            break
        fi
    done <<< "$gpu_info"
    
    if $all_free; then
        echo "$(date '+%H:%M:%S') - All GPUs free! Launching jobs..."
        break
    fi
    
    sleep $CHECK_INTERVAL
done

uv pip install vllm==0.11.2
# Launch fisher
# VERSION=fisher_v00.03 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_fisher_v00.03.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

# Launch vanilla SFT
VERSION=v00.16 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_v00.16.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.17 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_v00.17.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.18 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_v00.18.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.19 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_v00.19.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.20 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_science/config_distill_v00.20.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml


# launch evals
uv pip install vllm==0.10.1
uv pip uninstall flashinfer-python
uv pip install more_itertools syllapy "spacy[ja,ko,th]>=3.8.0" emoji "numpy==2.2"

VERSION=main

MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.16
bash eval_commands/gpqa.sh $MODEL $VERSION

MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.17
bash eval_commands/gpqa.sh $MODEL $VERSION

MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.18
bash eval_commands/gpqa.sh $MODEL $VERSION

MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.19
bash eval_commands/gpqa.sh $MODEL $VERSION

MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.20
bash eval_commands/gpqa.sh $MODEL $VERSION


# bash eval_commands/math_500.sh $MODEL $VERSION

# bash eval_commands/lcb.sh $MODEL $VERSION



uv pip install vllm==0.11.2
# uv run fisher_testbed/vllm_inference.py -m Neelectric/Llama-3.1-8B-Instruct_SFT_sciencev00.15 -mm 4096