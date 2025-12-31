### By Neel Rajani, 13.04.25
### Quick utility scripts

from transformers import AutoModelForCausalLM
from huggingface_hub import HfApi
from tqdm import tqdm
import subprocess
import os

def list_revisions(model_id: str) -> list[str]:
  """Returns all revisions of a model from the hub."""
  api = HfApi()
  refs = api.list_repo_refs(model_id)
  branch_names = [branch.name for branch in refs.branches]
  revisions = branch_names[1:] 
  revisions = sorted(revisions)
  return revisions

def download_all_revisions_fast(model_id):
  revisions = list_revisions(model_id)
  print(revisions)
  command = [
      "HF_HUB_ENABLE_HF_TRANSFER=1",
      "huggingface-cli",
      "download",
      model_id,
      "--cache-dir",
      "data/",
      "--max-workers",
      "32",
      "--revision",
      "placeholder_rev"
  ]

  revision_paths = {}
  for revision in tqdm(revisions, dynamic_ncols=True):
    command[-1] = revision
    result = subprocess.run(" ".join(command), shell=True, check=True,
                          capture_output=True, text=True)
    # The download path is typically printed in the output
    download_path = result.stdout.strip().split('\n')[-1]
    revision_paths[revision] = download_path

  return revision_paths

def run_evals_on_revisions(revision_paths: dict[str, str],
                           benchmark: str = "math_500",
                           num_gpus: int = 1,
                           num_toks: int = 4096,
                           output_dir: str = "data/auto_evals/"):
  """Runs lighteval benchmark on each revision with its downloaded path."""

  for revision, model_path in tqdm(revision_paths.items(), desc="Running evals"):
    print(f"\nEvaluating {revision} at {model_path}")

    # Set up model args similar to math_500.sh
    model_args = (
        f"model_name={model_path},"
        f"revision=\"{revision}\","
        f"dtype=bfloat16,"
        f"data_parallel_size={num_gpus},"
        f"max_model_length={num_toks},"
        f"gpu_memory_utilization=0.9,"
        f"generation_parameters={{max_new_tokens:{num_toks},temperature:0.6,top_p:0.95}}"
    )

    # Construct the lighteval command (updated for lighteval 0.13.0)
    env_vars = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
    }

    command = [
        "lighteval",
        "vllm",
        model_args,
        benchmark,
        "--output-dir",
        output_dir
    ]

    # Set environment variables and run
    env = os.environ.copy()
    env.update(env_vars)

    try:
      result = subprocess.run(command, env=env, check=True,
                            capture_output=True, text=True)
      print(f"✓ Completed eval for {revision}")
    except subprocess.CalledProcessError as e:
      print(f"✗ Failed eval for {revision}: {e}")
      print(f"stderr: {e.stderr}")

if __name__ == '__main__':
    model_id = "Neelectric/Llama-3.1-8B-Instruct_GRPO_Math-220kv00.10"
    revisions = list_revisions(model_id)
    print(revisions)
    revision_paths = download_all_revisions_fast(
        model_id=model_id
        )
    run_evals_on_revisions(revision_paths=revision_paths, num_gpus=5)