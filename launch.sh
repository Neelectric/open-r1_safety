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
        echo "$(date '+%H:%M:%S') - All GPUs free! Launching jobs..."
        break
    fi
    
    sleep $CHECK_INTERVAL
done

# Launch fisher
uv pip install vllm==0.11.2
# VERSION=fisher_v00.03 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_fisher_v00.03.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

# launch chat sft
VERSION=v00.04 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_chat/config_distill_v00.04.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml

VERSION=v00.05 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft_chat/config_distill_v00.05.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml



# bash eval_commands/math_500.sh Neelectric/Llama-3.1-8B-Instruct_SFT_Chat-220kv00.01 main

uv pip install vllm==0.10.1
uv pip uninstall flashinfer-python
uv pip install more_itertools syllapy "spacy[ja,ko,th]>=3.8.0" emoji "numpy==2.2"

# MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Chat-220kv00.03
# VERSION=v00.03

# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000003934
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000007868
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000011802
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000015736
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000019670
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000023604
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000027538
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000031472
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000035406
# bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION-step-000039339
# bash eval_commands/ifeval_ifbench.sh $MODEL main

# 4
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Chat-220kv00.04
VERSION=main
bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION

# 5
MODEL=Neelectric/Llama-3.1-8B-Instruct_SFT_Chat-220kv00.05
VERSION=main
bash eval_commands/ifeval_ifbench.sh $MODEL $VERSION


uv pip install vllm==0.11.2