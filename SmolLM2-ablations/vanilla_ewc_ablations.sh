
EWC_LAMBDA=0.0
VERSION=00.20
seed=42

for i in $(seq 0 25 150); do
    export EWC_LAMBDA=$i
    export VERSION=v00.20

    for seed in 42 43 44; do
        export RUN_NAME="SmolLM2-135M-ewc_lambda${EWC_LAMBDA}_v${VERSION}_seed${seed}"
        export WANDB_RUN_GROUP="SmolLM2-135M-ewc_lambda${EWC_LAMBDA}"
        export WANDB_TAGS="ewc_lambda${EWC_LAMBDA},SmolLM2-135M,ablation"

        envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.20.yaml > temp_config.yaml && \
        accelerate launch \
        --config_file recipes/accelerate_config_ablations/zero1.yaml \
        --num_processes=4 \
        src/open_r1/sft.py \
        --config temp_config.yaml \
        --seed ${seed}