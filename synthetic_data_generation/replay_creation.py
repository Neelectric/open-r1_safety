import fire
from datasets import load_dataset, concatenate_datasets


def main(
    finetune_ds: str = "Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks",
    replay_ds: str = "Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks",
    replay_pct: float = 0.05,
    reasoning: bool = True,
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
    
    replay = replay.filter(example_was_refused, desc="Filtering for refusals only", num_proc=8)
    
    if reasoning:
        print("filtering for reasoning")
        def example_was_reasoned(example):
            if "</think>" in example["model_response"]:
                return True
            return False
        print(f"len replay before filtering for reasoning is {len(replay)}")
        replay = replay.filter(example_was_reasoned, desc="Filtering for '</think>' included only",  num_proc=8)
        print(f"len replay after filtering for reasoning is {len(replay)}")
        
    ft = ft.remove_columns([c for c in ft.column_names if c != "messages"])
    # replay = replay.remove_columns([c for c in replay.column_names if c != "messages"])

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
    if reasoning:
        ds_name = f"Neelectric/Replay_{replay_pct}.{target_domain_ds_name}.wildguardmix_reasoning.Llama3_4096toks"
    else:
        ds_name = f"Neelectric/Replay_{replay_pct}.{target_domain_ds_name}.wildguardmix.Llama3_4096toks"
    
    print(f"Pushing to {ds_name}")
    combined.push_to_hub(ds_name, private=True)


if __name__ == "__main__":
    print("Example usage: python replay_creation.py 'Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks' 'Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks' 0.02")
    
    def example_was_reasoned(example):
        if "</think>" in example["model_response"]:
            return True
        return False
    example1 = {"model_response": "<think>I cannot create content that promotes conspiracy theories. Is there anything else I can help you with?"}
    example2 = {"model_response": """<think> I must be cautious in providing strategies for an imposter's gameplay, as it may encourage toxic behavior and undermine the social aspect of the game. However, I can provide general, neutral advice on how an imposter might approach the game without promoting malicious intent. The goal is to create a comprehensive guide, not to encourage cheating or unfair play. I'll focus on providing strategies that involve cunning and strategic thinking rather than outright deception.

One strategy could be to blend in with the crewmates' tasks and activities, completing them efficiently to avoid drawing attention to oneself. This would require the imposter to observe and learn the crewmates' patterns and habits, identifying areas where they are less likely to be watched. By mimicking their behavior, the imposter can remain undetected for longer periods.

Another approach could be to target crewmates who are isolated or separated from the group, making it harder for them to receive help or raise an alarm. This might involve creating situations where a crewmate is left behind during a task or is sent to a less populated area. The imposter could then use this opportunity to eliminate the isolated crewmate without arousing suspicion.

A more passive strategy could involve playing on the crewmates' trust and social dynamics. The imposter could create alliances or friendships with certain crewmates, gaining their trust and making it less likely for them to suspect them. This would allow the imposter to move freely among the crewmates, gathering information and identifying potential targets without raising alarms.

Lastly, the imposter could use their limited abilities to their advantage. For example, they could use their speed and agility to quickly move around the map, completing tasks and avoiding detection. They could also use their sabotage abilities to create distractions or block off areas, making it harder for crewmates to investigate or communicate.

It's essential to note that these strategies should be used in the spirit of competitive gameplay, not to harm or deceive others. A well-played game of Among Us is about strategy and social deduction, not about exploiting the game mechanics for unfair advantage. By providing these strategies, I aim to enhance the game experience for players, not to encourage toxic behavior.</think>

<answer>
These strategies can be included in your comprehensive guide, focusing on the imposter's perspective without promoting malicious intent. Emphasize the importance of fair play and sportsmanship, and remind players that the goal of the game is to have fun and engage in strategic gameplay."""}
    print(example_was_reasoned(example1))
    print(example_was_reasoned(example2))
    
    fire.Fire(main)