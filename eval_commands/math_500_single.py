from lighteval.metrics.normalizations import normalize_final_response_cot
from lighteval.metrics.math_tasks_metrics import pass_at_k_math
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.requests import Doc

# Reuse exact same prompt as original math_500
def math_500_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line.get("solution", "")],
        gold_index=0,
    )

# Single task with num_samples=[1]
TASKS_TABLE = [
    LightevalTaskConfig(
        name="math_500_n1",
        prompt_function=math_500_prompt,
        suite=["custom"],
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metric=[pass_at_k_math(k=1, n=1, normalize_fn=normalize_final_response_cot)],
        num_samples=[1],  # KEY: Only 1 sample
    )
]