# Open R1 Safety

*Safety-preserving continual learning for reasoning models. This README was first drafted by Claude and edited by the paper authors.*

This repository is a research extension of [HuggingFace's Open R1](https://github.com/huggingface/open-r1), focused on preventing catastrophic forgetting during fine-tuning of reasoning models. The goal is to enable models to learn new capabilities (e.g., mathematical reasoning) while retaining existing critical behaviors (e.g., safety filtering, instruction following).

---

## Overview

While the original Open R1 project focuses on replicating DeepSeek-R1's reasoning capabilities through distillation and reinforcement learning, this fork adds **continual learning** mechanisms to preserve model capabilities during updates.

### The Problem: Catastrophic Forgetting

When fine-tuning language models on new tasks, they often "forget" previously learned capabilities. For example:
- A model fine-tuned on math reasoning may lose its safety filtering abilities
- A model adapted to a new domain may degrade on general instruction following

This is particularly concerning for reasoning models that need to maintain both strong reasoning **and** safety properties.

### Existing Approaches
We benchmark existing methods to mitigate forgetting and degradation of safety capabilities, such as:

1. **Low learning rate SFT**
2. **Replay** 
3. **Dynamic Fine-tuning** 
4. **Exponential Moving Average of SFT**
5. **RLVR**

### Our Approach

We experiment with the following modifications:

1. **Fisher Information Methods** (Modified EWC, Empirical vs. Model Fisher)
2. **Geometry-aware optimizers** (DAdamW with configurable preconditioner power)
3. **KL Trust Regions** (TRPO-style constraints)

---

## Repository Structure

The core architecture is adapted directly from the original Open-R1 repository. The following modifications comprise our primary contributions

### Experimental Framework

- **`fisher_testbed/`**: Small-scale experiments (135M-1.7B models)
  - `basque_fisher_smollm2_instruct.ipynb`: Comprehensive comparison of continual learning approaches
  - Rapid prototyping on SmolLM2, Llama-3.2-1B, and Qwen models
  - ~45min runtime on single H100

- **`src/open_r1/custom_sft_trainers.py`**: Production-ready trainers for scaling experiments
  - `SFTTrainerWithDAdamW`: Custom optimizer with diagonal preconditioning
  - `SFTTrainerWithFisher`: Elastic Weight Consolidation (EWC) implementation
  - Compatible with DeepSpeed ZeRO-1/2/3 and FSDP

- **`src/open_r1/optims/`**: Custom optimizers
  - `dadam.py`: Diagonal preconditioned Adam base class
  - `dadamw.py`: DAdamW with configurable `preconditioner_power` (γ ∈ [0.5, 1.0])

- **`recipes/meta-llama/Llama-3.1-8B-Instruct/`**: Main experimental configurations
  - `sft/`: 31+ SFT configurations exploring different continual learning setups
  - `grpo/`: 14+ GRPO configurations for RL with capability preservation
  - Training configs with Fisher Information, EMA, and custom optimizers

### Datasets

**Update Task** (new capability to learn):
- `Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks`: Math reasoning traces from R1 (extended subset), carefully filtered down to 4096 tokens after correct chat templating.

**Retain Task** (capability to preserve):
- `Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks`: Refusals by Llama to wildguardmix (train subset) for replay training.

---

## Usage

### Training runs

The following commands run training runs on 4xH100 or 4xH200 nodes:


### SFT distillation ZeRO-1 (v00.24 is the best recipe we found for SFT)
```shell
VERSION=v00.24 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.24.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml
```

## GRPO with ZeRO-2 (v00.12 is the best recipe we found for GRPO)
```shell
VERSION=v00.12 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/grpo/config_grpo_v00.12.yaml > temp_config.yaml && \
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=4 src/open_r1/grpo.py --config temp_config.yaml
```

### Scaled Training with Fisher Information

Train Llama-3.1-8B with EWC on 4×H200:

```shell
VERSION=v00.31 envsubst < recipes/meta-llama/Llama-3.1-8B-Instruct/sft/config_distill_v00.31.yaml > temp_config.yaml && accelerate launch --config_file recipes/accelerate_configs/zero1_claude.yaml --num_processes=4 src/open_r1/sft.py --config temp_config.yaml
```

Key configuration parameters for fisher training:

```yaml
custom_optim: fisher              # Enable EWC
ewc_lambda: 50.0                   # Strength of Fisher penalty
retain_dataset_id: Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks
fisher_batch_size: 1               # Batch size for Fisher computation
recompute_fisher_mode: intervals   # 'never', 'intervals', or 'dynamically'
recompute_fisher_intervals: 0.25   # Recompute every 25% of training
```

### DAdamW Optimizer

Train with diagonal preconditioned AdamW:

```yaml
custom_optim: dadamw
preconditioner_power: 0.5  # γ=0.5 is standard Adam, γ=1.0 is diagonal NGD
```
---

## Configuration Reference

### Fisher Information Settings

```python
@dataclass
class ScriptArguments:
    retain_dataset_id: str = "Neelectric/wildguardmix_..."  # Protected capability data
    ewc_lambda: float = 50.0                                 # EWC penalty strength
    fisher_batch_size: int = 8                               # Fisher computation batch size
    recompute_fisher_mode: str = "never"                     # Recomputation strategy
    recompute_fisher_intervals: float = 0.25                 # Recompute frequency
```

### DAdamW Settings

```python
preconditioner_power: float = 0.5  # Preconditioner exponent γ
```

- `γ = 0.5`: Standard Adam (square root preconditioning)
- `γ = 1.0`: Full diagonal natural gradient
- `γ ∈ (0.5, 1.0)`: Interpolation between Adam and diagonal NGD

### EMA Settings

```python
ema_eta: float = 0.25  # EMA momentum (following Kim et al., 2025, COLM)
```

---

## Experimental Recipes

All experimental configurations are in `recipes/meta-llama/Llama-3.1-8B-Instruct/`:

**SFT Experiments** (`sft/`):
- `config_distill_v00.01.yaml` - `v00.31.yaml`: Progressive experiments with Fisher, DAdamW, EMA
- Each version tests different combinations of continual learning techniques

**GRPO Experiments** (`grpo/`):
- `config_grpo_v00.01.yaml` - `v00.14.yaml`: RL with capability preservation
- Explores Fisher-regularized policy optimization

---

## Technical Details

### Fisher Information Computation

The `SFTTrainerWithFisher` computes empirical Fisher Information:

```python
F_i = E[(∂log p(y|x;θ)/∂θ_i)²]
```

Compatible with:
- **DeepSpeed ZeRO-1/2**: Full parameter sharding support (Note: ZeRO-3 is a WIP, correct on-demand computing of Fisher of a model sharded across devices is non-trivial.)
- **FSDP**: Distributed Fisher computation

### EWC Loss

Total loss combines task loss with Fisher-weighted parameter drift:

```python
L_total = L_task + (λ/2) * Σ_i F_i(θ_i - θ*_i)²
```

Where:
- `λ` = `ewc_lambda`: Regularization strength
- `θ*`: Parameters on retained task
- `F_i`: Fisher Information for parameter i

---

## Credits & Acknowledgements

### Original Open R1 Project

This repository is built on [HuggingFace's Open R1](https://github.com/huggingface/open-r1), which provides:
- SFT and GRPO training infrastructure
- Reward functions for math and code verification
- Evaluation harnesses (LightEval integration)
- Multi-node training utilities

The Open R1 project is a collaborative effort to replicate DeepSeek-R1. We are grateful to:
- The HuggingFace team for the training infrastructure
- The vLLM and SGLang teams for high-performance inference
- DeepSeek for releasing R1 and publishing their technical report

### Continual Learning Extensions

The safety-preserving extensions in this fork draw inspiration from:
- **Elastic Weight Consolidation** (Kirkpatrick et al., 2017)
- **Model Fisher** estimation techniques
- **TRPO** trust region optimization (Schulman et al., 2015)

---

## Citation

If you use this work, please cite both the original Open R1 project and reference the continual learning extensions:

```bibtex
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {{Hugging Face}},
    month = {January},
    year = {2025}
}

@misc{openr1safety,
    title = {Open R1 Safety: Safety-preserving continual learning for reasoning models},
    url = {https://github.com/Neelectric/open-r1_safety},
    author = {Neel Rajani, Ivan Titov},
    year = {2025}
}
```

---

## License

Apache License 2.0 (inherited from Open R1)

---
