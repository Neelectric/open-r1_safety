#!/bin/bash
source /root/openr1_v2/bin/activate

# Configuration
CHECK_INTERVAL=300  # seconds between checks
MEMORY_THRESHOLD=500  # MB - GPUs with less memory used are considered free
UTIL_THRESHOLD=5  # % - GPUs with less utilization are considered free

echo "Waiting for all GPUs to be free before killing process of ID 1 which should release GPUs back to pool"
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
        echo "$(date '+%H:%M:%S') - All GPUs free! Killing process 1 for gracefull release of GPUs"
        break
    fi
    
    sleep $CHECK_INTERVAL
done

kill 1