"""Tests for bond feasibility scoring."""

import pytest
from fragmentretro.bond_feasibility import (
    DEFAULT_FEASIBILITY,
    get_bond_feasibility,
    get_solution_bond_feasibility,
)


def test_known_bond_types():
    """Well-known reactions should have high feasibility."""
    assert get_bond_feasibility("1", "5") == 1.0   # amide coupling
    assert get_bond_feasibility("1", "3") == 1.0   # ester formation
    assert get_bond_feasibility("5", "12") == 1.0  # sulfonamide


def test_symmetric_lookup():
    """Order of labels shouldn't matter."""
    assert get_bond_feasibility("1", "5") == get_bond_feasibility("5", "1")
    assert get_bond_feasibility("14", "16") == get_bond_feasibility("16", "14")


def test_unknown_bond_type():
    """Unknown combinations should return default."""
    assert get_bond_feasibility("999", "888") == DEFAULT_FEASIBILITY


def test_ring_fusion_challenging():
    """Ring fusion bonds should have low feasibility."""
    assert get_bond_feasibility("18", "19") <= 0.4
    assert get_bond_feasibility("182", "192") <= 0.3


def test_solution_no_bonds():
    """Single fragment (target is a BB) should return 1.0."""
    assert get_solution_bond_feasibility([]) == 1.0


def test_solution_average():
    """Should return average of individual bond scores."""
    bonds = [("1", "5"), ("14", "16")]
    score = get_solution_bond_feasibility(bonds)
    expected = (get_bond_feasibility("1", "5") + get_bond_feasibility("14", "16")) / 2
    assert abs(score - expected) < 1e-6


if __name__ == "__main__":
    pytest.main()
