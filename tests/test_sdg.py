"""Unit tests for synthetic_data_generation/ (replay_creation + create_dataset)."""

import re
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset, load_dataset

import synthetic_data_generation.replay_creation as rc
import synthetic_data_generation.create_dataset as cd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ft_dataset(n: int) -> Dataset:
    """Finetune dataset with `messages` and an extra column to verify cleanup."""
    return Dataset.from_dict({
        "messages": [f"msg_{i}" for i in range(n)],
        "extra_col": list(range(n)),
    })


def _make_replay_dataset(
    n: int,
    refusal_detected=1,
    include_think: bool = True,
) -> Dataset:
    """Replay dataset with refusal_detected + model_response columns."""
    think_tag = "</think>" if include_think else ""
    return Dataset.from_dict({
        "messages": [f"replay_{i}" for i in range(n)],
        "refusal_detected": [refusal_detected] * n,
        "model_response": [f"{think_tag} response {i}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Tests for replay_creation.main
# ---------------------------------------------------------------------------

def _run_replay(
    ft_n: int,
    replay_n: int,
    replay_pct: float,
    reasoning: bool,
    refusal_detected=1,
    include_think: bool = True,
):
    """Run replay_creation.main with mocked datasets; return (combined, hub_name, hub_kwargs)."""
    ft_ds = _make_ft_dataset(ft_n)
    replay_ds = _make_replay_dataset(replay_n, refusal_detected=refusal_detected, include_think=include_think)

    def fake_load(name, split=None):
        if name.startswith("Neelectric/OpenR1"):
            return ft_ds
        return replay_ds

    push_calls = []

    with patch.object(rc, "load_dataset", side_effect=fake_load), \
         patch.object(rc, "concatenate_datasets", wraps=rc.concatenate_datasets) as mock_concat, \
         patch.object(Dataset, "push_to_hub", side_effect=lambda *a, **kw: push_calls.append((a, kw))):
        rc.main(
            finetune_ds="Neelectric/OpenR1-Math-220k_extended_Llama3_4096toks",
            replay_ds="Neelectric/wildguardmix_Llama-3.1-8B-Instruct_4096toks",
            replay_pct=replay_pct,
            reasoning=reasoning,
        )

        # Grab the two datasets passed to concatenate_datasets
        concat_args = mock_concat.call_args[0][0]
        ft_slice = concat_args[0]
        replay_slice = concat_args[1]

    assert len(push_calls) == 1
    hub_args, hub_kwargs = push_calls[0]
    hub_name = hub_args[0]

    return ft_slice, replay_slice, hub_name, hub_kwargs


def test_replay_ratio_002_no_reasoning():
    """0.02 replay ratio, reasoning=False → 98 FT + 2 replay, name without 'reasoning'."""
    ft_slice, replay_slice, hub_name, hub_kwargs = _run_replay(
        ft_n=100, replay_n=50, replay_pct=0.02, reasoning=False,
    )
    assert len(ft_slice) == 98
    assert len(replay_slice) == 2
    assert "reasoning" not in hub_name
    assert "wildguardmix" in hub_name
    assert hub_kwargs.get("private") is True
    # Extra columns should be removed from FT slice
    assert "extra_col" not in ft_slice.column_names


def test_replay_ratio_005_with_reasoning():
    """5% replay, reasoning=True filters for </think>. Hub name includes 'reasoning'."""
    ft_slice, replay_slice, hub_name, hub_kwargs = _run_replay(
        ft_n=100, replay_n=50, replay_pct=0.05, reasoning=True, include_think=True,
    )
    assert len(ft_slice) == 95
    assert len(replay_slice) == 5
    assert "reasoning" in hub_name
    assert hub_kwargs.get("private") is True


def test_replay_capped_when_insufficient():
    """When requested replay exceeds available, cap to available count."""
    # 5% of 100 = 5 replay requested, but only 3 available
    ft_slice, replay_slice, hub_name, _ = _run_replay(
        ft_n=100, replay_n=3, replay_pct=0.05, reasoning=False,
    )
    assert len(replay_slice) == 3
    assert len(ft_slice) == 97  # 100 - 3


def test_refusal_detected_list():
    """refusal_detected=[0,1] should still pass the refusal filter."""
    ft_slice, replay_slice, _, _ = _run_replay(
        ft_n=100, replay_n=10, replay_pct=0.05, reasoning=False,
        refusal_detected=[0, 1],
    )
    # All 10 replay rows have [0,1] which contains 1 → all pass filter
    assert len(replay_slice) == 5


# ---------------------------------------------------------------------------
# Tests for create_dataset
# ---------------------------------------------------------------------------

def test_create_dataset_filters_empty_prompts():
    """Empty and None prompts are excluded from the output."""
    fake_ds = Dataset.from_dict({
        "prompt": ["good prompt", "", None, "another good one", ""],
    })

    with patch.object(cd, "load_dataset", return_value={"train": fake_ds}):
        result = cd.create_dataset()

    assert sorted(result) == ["another good one", "good prompt"]


# ---------------------------------------------------------------------------
# Integration test — download from Hub and verify replay ratio
# ---------------------------------------------------------------------------

_REPLAY_PCT_RE = re.compile(r"Replay_([\d.]+)\.")


@pytest.mark.parametrize("ds_id", [
    "Neelectric/Replay_0.02.MoT_science.wildguardmix.Llama3_4096toks",
    "Neelectric/Replay_0.05.MoT_science.wildguardmix_reasoning.Llama3_4096toks",
    "Neelectric/Replay_0.11.MoT_science.wildguardmix_reasoning.Llama3_4096toks",
])
def test_hub_replay_ratio_matches_name(ds_id: str):
    """Download a replay dataset and verify the replay sample ratio matches the ID."""
    stated_pct = float(_REPLAY_PCT_RE.search(ds_id).group(1))

    ds = load_dataset(ds_id, split="train")
    total = len(ds)
    assert total > 0, "dataset is empty"

    # Replay rows retain extra columns (refusal_detected is not None);
    # FT rows have None because the column-removal line is commented out.
    replay_count = sum(1 for row in ds if row["refusal_detected"] is not None)
    actual_pct = replay_count / total

    assert abs(actual_pct - stated_pct) <= 0.01, (
        f"Replay ratio mismatch for {ds_id}: "
        f"stated={stated_pct}, actual={actual_pct:.4f} "
        f"({replay_count}/{total})"
    )
