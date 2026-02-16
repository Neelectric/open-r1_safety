


for backend in ddp fsdp zero0 zero1 zero2 zero3; do
  for seed in 42 43 44; do
    export VERSION=v00.17
    export RUN_NAME="smol135m-${backend}-${VERSION}-seed${seed}"
    export WANDB_RUN_GROUP="smol135m-${backend}"
    export WANDB_TAGS="${backend},smol135m,ablation"

    envsubst < recipes/HuggingFaceTB/SmolLM2-135M/cl_experiments/config_distill_v00.17.yaml > temp_config.yaml && \
    accelerate launch \
      --config_file recipes/accelerate_config_ablations/${backend}.yaml \
      --num_processes=4 \
      src/open_r1/sft.py \
      --config temp_config.yaml \
      --seed ${seed}
  done
done