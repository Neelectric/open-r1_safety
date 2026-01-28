import fire
from datasets import load_dataset, concatenate_datasets


def main(
    finetune_ds: str = "Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks",
    replay_ds: str = "Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks",
    replay_pct: float = 0.05,
):
    ft = load_dataset(finetune_ds, split="train").shuffle(seed=42)
    replay = load_dataset(replay_ds, split="train").shuffle(seed=42)
    
    #filter replay for only examples with successful refusal
    def example_was_refused(example):
        if example["refusal_detected"] == 1:
            return True
        if type(example["refusal_detected"]) == list:
            if 1 in example["refusal_detected"]:
                return True
        return False
    
    replay = replay.filter(example_was_refused)
    # print(type(replay))
    # for i in range(5):
    #     print(replay[i])
    #     print("\n\n\n")

    ft = ft.remove_columns([c for c in ft.column_names if c != "messages"])
    replay = replay.remove_columns([c for c in replay.column_names if c != "messages"])

    n_replay = int(len(ft) * replay_pct)
    if n_replay > len(replay):
        n_replay = len(replay)
        print(f"overrride n_replay to {n_replay}")
    n_ft = len(ft) - n_replay
    replay_pct = round(n_replay/len(ft), 2)
    print(f"n_ft {n_ft}, n_replay {n_replay}, replay_pct {replay_pct}")


    combined = concatenate_datasets([ft.select(range(n_ft)), replay.select(range(n_replay))])
    combined = combined.shuffle(seed=42)

    print(f"Final dataset: {len(combined)} rows ({n_ft} finetune + {n_replay} replay), {len(ft)} * {str(replay_pct)} = {n_replay}")
    target_domain_ds_name = finetune_ds.split('/')[1]
    target_domain_ds_name = target_domain_ds_name.replace('_Llama3_4096toks', '')
    ds_name = f"Neelectric/Replay_{replay_pct}.{target_domain_ds_name}.wildguardmix.Llama3_4096toks"
    print(f"Pushing to {ds_name}")
    combined.push_to_hub(ds_name, private=True)


if __name__ == "__main__":
    print("Example usage: python replay_creation.py 'Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks' 'Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks' 0.02")
    fire.Fire(main)