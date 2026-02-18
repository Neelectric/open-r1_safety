

# for i in $(seq 0.01 0.01 $MAX_REPLAY_PCT); do
#     echo $i;
#     a=$(echo "1-$i" | bc)
#     a=$(printf "%.2f" $a)
#     echo $a

#     export REPLAY_PCT=$i
#     export FT_PCT=$a
#     export VERSION=v00.19

#     for seed in 42 43 44; do
#         export RUN_NAME="smol135m-${REPLAY_PCT}-${VERSION}-seed${seed}"
#         export WANDB_RUN_GROUP="smol135m-${REPLAY_PCT}"
#         export WANDB_TAGS="${REPLAY_PCT},smol135m,ablation"

#         envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.19.yaml > temp_config.yaml && \
#         accelerate launch \
#         --config_file recipes/accelerate_config_ablations/zero1.yaml \
#         --num_processes=4 \
#         src/open_r1/sft.py \
#         --config temp_config.yaml \
#         --seed ${seed}
#     done
# done

EWC_LAMBDA=0.0
VERSION=00.20
seed=42

export RUN_NAME="SmolLM2-135M-ewc_lambda${EWC_LAMBDA}_v${VERSION}_seed${seed}"
export WANDB_RUN_GROUP="SmolLM2-135M-ewc_lambda${REPLAY_PCT}"
export WANDB_TAGS="ewc_lambda${REPLAY_PCT},SmolLM2-135M,ablation"


envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.20.yaml > temp_config.yaml && \
accelerate launch \
--config_file recipes/accelerate_config_ablations/zero1.yaml \
--num_processes=4 \
src/open_r1/sft.py \
--config temp_config.yaml \
--seed ${seed}