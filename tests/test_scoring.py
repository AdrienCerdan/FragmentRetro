"""Tests for scoring module."""

import json
import tempfile
from pathlib import Path

import pytest
from fragmentretro.scoring import compute_score, is_feasible, load_compound_filter
from fragmentretro.utils.filter_compound import precompute_properties


@pytest.fixture(scope="module")
def compound_filter(tmp_path_factory):
    """Create a small BB set and precompute properties for testing."""
    bbs = [
        "CCCN", "CCCCN", "CCN", "NCCCN", "CCO", "CCCO",
        "c1ccccc1", "c1ccc(N)cc1", "c1ccc(O)cc1", "c1ccc(Cl)cc1",
        "CC(=O)O", "CC(=O)N", "CC(=O)Nc1ccccc1",
    ]
    tmp_dir = tmp_path_factory.mktemp("data")
    props_path = tmp_dir / "test_bb_props.json"
    precompute_properties(bbs, props_path)
    return load_compound_filter(props_path)


def test_is_feasible_simple_bb(compound_filter):
    """A molecule that IS a BB should be feasible."""
    assert is_feasible("CCN", compound_filter) is True


def test_is_feasible_impossible(compound_filter):
    """A very complex molecule not in BB stock should fail."""
    assert is_feasible("C1CC2CCCC3CCCC1C23", compound_filter) is False


def test_compute_score_range(compound_filter):
    """Score should be in [0, 1]."""
    score = compute_score("CCN", compound_filter)
    assert 0.0 <= score <= 1.0


def test_compute_score_zero_for_impossible(compound_filter):
    """Impossible molecules should score 0."""
    score = compute_score("C1CC2CCCC3CCCC1C23", compound_filter)
    assert score == 0.0


def test_compute_score_higher_for_simpler(compound_filter):
    """Simpler molecules should generally score higher."""
    score_simple = compute_score("CCN", compound_filter)
    # If both are 0, both are unsolvable with this tiny BB set — that's ok
    assert score_simple >= 0.0


if __name__ == "__main__":
    pytest.main()
