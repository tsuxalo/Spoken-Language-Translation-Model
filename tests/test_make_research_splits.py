from __future__ import annotations

import random

import pytest

from experiments.make_research_splits import (
    build_leakage_components,
    select_validation_ids,
)


def _row(alignment_id: str, pair: str, rendition: int = 0) -> dict[str, object]:
    return {
        "alignment_id": alignment_id,
        "utterance_id": f"{alignment_id}-{rendition}",
        "hausa_gold": f"hausa {pair}",
        "english_ref": f"english {pair}",
        "speaker_id": f"speaker-{rendition}",
        "wer": 0.0,
    }


def test_unique_pairs_preserve_authoritative_seeded_assignment() -> None:
    groups = {f"a{i}": [_row(f"a{i}", str(i))] for i in range(10)}
    expected = sorted(groups)
    random.Random(42).shuffle(expected)
    selected, stats = select_validation_ids(groups, val_fraction=0.2, seed=42)
    assert selected == set(expected[:2])
    assert stats["target_validation_alignment_ids"] == 2


def test_duplicate_pair_alignment_ids_remain_together() -> None:
    groups = {
        "a": [_row("a", "duplicate")],
        "b": [_row("b", "duplicate")],
        "c": [_row("c", "unique-c")],
        "d": [_row("d", "unique-d")],
    }
    selected, stats = select_validation_ids(groups, val_fraction=0.5, seed=7)
    assert ("a" in selected) == ("b" in selected)
    assert stats["cross_alignment_text_pair_collisions"] == 1
    assert stats["max_component_alignment_ids"] == 2


def test_inconsistent_alignment_pair_is_rejected() -> None:
    groups = {"a": [_row("a", "one", 0), _row("a", "two", 1)]}
    with pytest.raises(ValueError, match="maps to 2 bilingual pairs"):
        build_leakage_components(groups)


def test_seeded_selection_is_deterministic_and_conservative() -> None:
    groups = {f"a{i}": [_row(f"a{i}", str(i))] for i in range(20)}
    first, stats = select_validation_ids(groups, val_fraction=0.1, seed=42)
    second, _ = select_validation_ids(groups, val_fraction=0.1, seed=42)
    train = set(groups) - first
    assert first == second
    assert len(first) == stats["target_validation_alignment_ids"] == 2
    assert first.isdisjoint(train)
    train_pairs = {
        (groups[key][0]["hausa_gold"], groups[key][0]["english_ref"])
        for key in train
    }
    val_pairs = {
        (groups[key][0]["hausa_gold"], groups[key][0]["english_ref"])
        for key in first
    }
    assert train_pairs.isdisjoint(val_pairs)
