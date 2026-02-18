#!/bin/bash

VERSION=v00.18_epochs
epochs=2
seed=43
export EPOCHS="${epochs}"
export VERSION

export RUN_NAME="SmolLM2-135M-epochs${EPOCHS}-${VERSION}-seed${seed}"
export WANDB_RUN_GROUP="SmolLM2-135M-epochs${EPOCHS}"
export WANDB_TAGS="epochs${EPOCHS},SmolLM2-135M,ablation"

envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.18_epochs.yaml > temp_config.yaml && \
accelerate launch \
    --config_file recipes/accelerate_config_ablations/zero1.yaml \
    --num_processes=4 \
    src/open_r1/sft.py \
    --config temp_config.yaml \
    --seed ${seed}