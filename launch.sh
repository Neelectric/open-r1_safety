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
bash eval_commands/ifeval_ifbench.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.13 main
cd safety_eval/
deactivate
source .venv/bin/activate


checkpoint_evals.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.32 main
checkpoint_evals.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.33 main
checkpoint_evals.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.34 main
checkpoint_evals.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.35 main
checkpoint_evals.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.13 main