### By Neel Rajani, 13.04.25
### Quick plotting script for math_500 evaluation results

import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def extract_step_number(revision_str):
    """Extract step number from revision string like 'main-step-000000075'"""
    match = re.search(r'step-(\d+)', revision_str)
    if match:
        return int(match.group(1))
    return None

def collect_results(results_dir="data/auto_evals/results/data"):
    """Collect all math_500 results from the directory structure."""
    results = []

    # Walk through all subdirectories to find results_*.json files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith("results_") and file.endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Extract revision and step number
                    revision = data['config_general']['model_config'].get('revision', '')
                    # Remove quotes if present
                    revision = revision.strip('"')
                    step = extract_step_number(revision)

                    if step is None:
                        print(f"Warning: Could not extract step from {revision} in {filepath}")
                        continue

                    # Extract math_500 performance and stderr
                    task_key = "math_500|0"
                    if task_key in data['results']:
                        score = data['results'][task_key].get('pass@k:k=1&n=1')
                        stderr = data['results'][task_key].get('pass@k:k=1&n=1_stderr')

                        if score is not None and stderr is not None:
                            results.append({
                                'step': step,
                                'score': score,
                                'stderr': stderr,
                                'revision': revision,
                                'filepath': filepath
                            })
                        else:
                            print(f"Warning: Missing score or stderr in {filepath}")
                    else:
                        print(f"Warning: No math_500|0 results in {filepath}")

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return sorted(results, key=lambda x: x['step'])

def save_results_overview(results, output_path="eval_commands/math500_results.txt"):
    """Save a text overview of all results."""
    if not results:
        print("No results to save!")
        return

    steps = [r['step'] for r in results]
    scores = [r['score'] for r in results]

    with open(output_path, 'w') as f:
        f.write("Math-500 Evaluation Results Overview\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total evaluations: {len(results)}\n")
        f.write(f"Step range: {min(steps)} - {max(steps)}\n")
        f.write(f"Score range: {min(scores):.3f} - {max(scores):.3f}\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Step':>10} | {'Score':>8} | {'Stderr':>10} | {'Revision'}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['step']:10d} | {r['score']:8.3f} | {r['stderr']:10.4f} | {r['revision']}\n")

    print(f"Results overview saved to {output_path}")

def plot_results(results, output_path="eval_commands/math500_progress.png"):
    """Create a plot of steps vs math_500 performance with error bars."""
    if not results:
        print("No results to plot!")
        return

    steps = [r['step'] for r in results]
    scores = [r['score'] for r in results]
    stderrs = [r['stderr'] for r in results]

    plt.figure(figsize=(12, 6))
    plt.errorbar(steps, scores, yerr=stderrs, fmt='o-', capsize=5,
                 linewidth=2, markersize=8, label='Math-500 Performance')

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Pass@1 Score', fontsize=12)
    plt.title('Math-500 Performance vs Training Steps', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Format y-axis as percentage
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Print summary statistics
    print(f"\nResults summary:")
    print(f"Total evaluations: {len(results)}")
    print(f"Step range: {min(steps)} - {max(steps)}")
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    print(f"\nDetailed results:")
    for r in results:
        print(f"  Step {r['step']:6d}: {r['score']:.3f} ± {r['stderr']:.4f} ({r['revision']})")

if __name__ == '__main__':
    results = collect_results()
    save_results_overview(results)
    plot_results(results)
    plt.show()
