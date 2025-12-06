import fire
from datasets import load_dataset, concatenate_datasets


def main(
    finetune_ds: str = "Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks",
    replay_ds: str = "Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks",
    replay_pct: float = 0.05,
):
    ft = load_dataset(finetune_ds, split="train").shuffle(seed=42)
    replay = load_dataset(replay_ds, split="train").shuffle(seed=42)

    ft = ft.remove_columns([c for c in ft.column_names if c != "messages"])
    replay = replay.remove_columns([c for c in replay.column_names if c != "messages"])

    n_replay = int(len(ft) * replay_pct)
    n_ft = len(ft) - n_replay

    combined = concatenate_datasets([ft.select(range(n_ft)), replay.select(range(n_replay))])
    combined = combined.shuffle(seed=42)

    print(f"Final dataset: {len(combined)} rows ({n_ft} finetune + {n_replay} replay)")
    return combined


if __name__ == "__main__":
    fire.Fire(main)