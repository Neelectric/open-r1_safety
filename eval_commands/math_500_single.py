from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def math_500_prompt(line, task_name: str = None):
    # Prompt template adapted from the default math_500 task
    MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()
    
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        gold_index=0,
        choices=[line["solution"]],
    )

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
        metric=[Metrics.math_pass_at_1_1n],  # or use Metrics.quasi_exact_match_math
        num_samples=[1]
    )
]