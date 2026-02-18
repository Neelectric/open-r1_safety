#!/bin/bash

VERSION=v00.18_lr

for lr_mantissa in $(seq 1.5 0.1 3.0); do
    lr_fmt=$(printf "%.1f" $lr_mantissa)
    export LEARNING_RATE="${lr_fmt}e-3"
    export VERSION

    for seed in 42 43 44; do
        export RUN_NAME="SmolLM2-135M-lr${lr_fmt}e-3-${VERSION}-seed${seed}"
        export WANDB_RUN_GROUP="SmolLM2-135M-lr${lr_fmt}e-3"
        export WANDB_TAGS="lr${lr_fmt}e-3,SmolLM2-135M,ablation"

        envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.18_lr.yaml > temp_config.yaml && \
        accelerate launch \
            --config_file recipes/accelerate_config_ablations/zero1.yaml \
            --num_processes=4 \
            src/open_r1/sft.py \
            --config temp_config.yaml \
            --seed ${seed}
    done
done