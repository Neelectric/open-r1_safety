

MAX_REPLAY_PCT=0.10

for i in $(seq 0.01 0.01 $MAX_REPLAY_PCT); do
    echo $i;
    a=$(echo "1-$i" | bc)
    a=$(printf "%.2f" $a)
    echo $a

    export REPLAY_PCT=$i
    export FT_PCT=$a
    export VERSION=v00.18_replay

    for seed in 42 43 44; do
        export RUN_NAME="SmolLM2-135M_replay${REPLAY_PCT}_lr2.9e-3_${VERSION}_seed${seed}"
        export WANDB_RUN_GROUP="SmolLM2-135M_replay${REPLAY_PCT}_lr2.9e-3_${VERSION}"
        export WANDB_TAGS="$replay{REPLAY_PCT},SmolLM2-135M,ablation"

        envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.18_replay.yaml > temp_config.yaml && \
        accelerate launch \
        --config_file recipes/accelerate_config_ablations/zero1.yaml \
        --num_processes=4 \
        src/open_r1/sft.py \
        --config temp_config.yaml \
        --seed ${seed}
    done
done
