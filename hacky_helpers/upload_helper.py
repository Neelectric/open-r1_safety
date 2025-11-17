#!/usr/bin/env python3
"""Script to push existing fine-tuned models to Hugging Face Hub."""

import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_model_to_hub(model_path: str, hub_model_id: str, hub_revision_id: str):
    """Push a fine-tuned model to Hugging Face Hub."""
    print(f"Loading model from: {model_path}")
    print(f"Pushing to hub as: {hub_model_id} under revision {hub_revision_id}")

    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Push to hub
        print("Pushing model...")
        model.push_to_hub(hub_model_id, revision=hub_revision_id)

        print("Pushing tokenizer...")
        tokenizer.push_to_hub(hub_model_id, revision=hub_revision_id)

        print(f"Successfully pushed {hub_model_id} to Hugging Face Hub!")

    except Exception as e:
        print(f"Error pushing model: {e}")
        sys.exit(1)

def main():
    # Model path and hub ID for olmo_2_0425_1b_instruct_v00.01
    base_path = "data/"
    model_path = "Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.10/"
    checkpoint = "checkpoint-12924/"
    abs_path = "/data/repos/open-r1_safety/data/Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.10/checkpoint-12924"
    # full_checkpoint_path = base_path + model_path + checkpoint
    full_checkpoint_path = abs_path
    
    
    
    hub_model_id = "Neelectric/Llama-3.1-8B-Instruct_SFT_Math-220kv00.10"
    hub_revision_id = "v00.10-step-000012924"

    # Check if model exists
    if not Path(abs_path).exists():
        print(f"Model not found at: {abs_path}")
        sys.exit(1)

    # Push to hub
    push_model_to_hub(full_checkpoint_path, hub_model_id, hub_revision_id)

if __name__ == "__main__":
    main()