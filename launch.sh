#!/bin/bash

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
        echo "$(date '+%H:%M:%S') - All GPUs free! Launching training..."
        break
    fi
    
    sleep $CHECK_INTERVAL
done

# Launch the command
VERSION=v00.32 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.32.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.33 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.33.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.34 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.34.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.35 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.35.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml