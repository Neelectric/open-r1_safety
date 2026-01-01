#!/usr/bin/env python3
### Run evaluation for v00.12 only
### Use this after v00.10 completes

import sys
sys.path.append('utils')
from utils import download_all_revisions_fast, run_evals_on_revisions

if __name__ == '__main__':
    model_id = "Neelectric/Llama-3.1-8B-Instruct_GRPO_Math-220kv00.12"

    print(f"\n{'='*80}")
    print(f"Starting evaluation for: {model_id}")
    print(f"{'='*80}\n")

    # Download all revisions
    print(f"Downloading revisions for {model_id}...")
    revision_paths = download_all_revisions_fast(model_id)

    # Run evaluations
    print(f"\nRunning evaluations for {model_id}...")
    run_evals_on_revisions(revision_paths=revision_paths, num_gpus=5)

    print(f"\n{'='*80}")
    print(f"✓ Completed all evaluations for {model_id}")
    print(f"{'='*80}")
