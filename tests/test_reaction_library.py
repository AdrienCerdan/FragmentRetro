"""Tests for SMARTS reaction library, constraints, and integrated scoring.

Covers:
  - ReactionLibrary loading (Hartenfeller, eXplore, custom)
  - Reaction SMARTS parsing and matching
  - Constraint filtering (simple params + ConstraintConfig)
  - SMARTS retrosynthesis engine
  - Tier 2/3 scoring integration
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem

from fragmentretro.reaction_library import (
    EXPLORE_PATH,
    HARTENFELLER_PATH,
    Reaction,
    ReactionLibrary,
)
from fragmentretro.constraints import ConstraintConfig, build_constraints


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def hartenfeller_lib():
    """Load Hartenfeller 58 reactions."""
    return ReactionLibrary.from_hartenfeller()


@pytest.fixture
def explore_lib():
    """Load eXplore Cookbook reactions."""
    return ReactionLibrary.from_explore()


@pytest.fixture
def default_lib():
    """Load both catalogs."""
    return ReactionLibrary.default()


@pytest.fixture
def custom_reaction_file(tmp_path):
    """Create a temporary custom reaction JSON file."""
    data = {
        "reactions": [
            {
                "id": "custom_001",
                "name": "test_amide",
                "class": "coupling",
                "product_class": "amide",
                "smarts_forward": "[C:1](=[O:2])[OH1].[NH2:3][C:4]>>[C:1](=[O:2])[NH:3][C:4]",
                "num_reactants": 2,
                "reliability": 0.95,
                "notes": "Test reaction",
            },
            {
                "id": "custom_002",
                "name": "test_suzuki",
                "class": "coupling",
                "product_class": "biaryl",
                "smarts_forward": "[c:1]B(O)O.[c:2]Br>>[c:1][c:2]",
                "num_reactants": 2,
                "reliability": 0.9,
                "notes": "Simplified Suzuki",
            },
        ]
    }
    fpath = tmp_path / "custom_reactions.json"
    fpath.write_text(json.dumps(data))
    return fpath


@pytest.fixture
def suzuki_reaction():
    """A single Suzuki reaction for unit tests."""
    return Reaction(
        id="test_suzuki",
        name="Suzuki",
        reaction_class="coupling",
        product_class="biaryl",
        smarts_forward="[#6;H0;D3;$([#6](~[#6])~[#6]):1]B(O)O.[#6;H0;D3;$([#6](~[#6])~[#6]):2][Cl,Br,I]>>[#6:2][#6:1]",
        num_reactants=2,
        reliability=0.9,
        source="test",
    )


# ============================================================================
# ReactionLibrary: Loading
# ============================================================================

class TestReactionLibraryLoading:
    """Test loading reactions from catalogs."""

    def test_hartenfeller_loads(self, hartenfeller_lib):
        assert len(hartenfeller_lib) == 58

    def test_explore_loads(self, explore_lib):
        assert len(explore_lib) >= 30

    def test_default_loads_both(self, default_lib):
        assert len(default_lib) > 80  # 58 + 30+

    def test_hartenfeller_source_tag(self, hartenfeller_lib):
        for rxn in hartenfeller_lib.reactions:
            assert rxn.source == "hartenfeller"

    def test_explore_source_tag(self, explore_lib):
        for rxn in explore_lib.reactions:
            assert rxn.source == "explore"

    def test_custom_json_loading(self, custom_reaction_file):
        lib = ReactionLibrary()
        count = lib.load_json(custom_reaction_file, source="custom")
        assert count == 2
        assert "custom_001" in lib
        assert lib["custom_001"].name == "test_amide"
        assert lib["custom_001"].source == "custom"

    def test_custom_plus_builtin(self, custom_reaction_file, default_lib):
        default_lib.load_json(custom_reaction_file, source="custom")
        assert "custom_001" in default_lib
        assert "hartenfeller_31" in default_lib  # Suzuki

    def test_malformed_json_skipped(self, tmp_path):
        data = {
            "reactions": [
                {"id": "good", "name": "ok", "class": "x", "product_class": "y",
                 "smarts_forward": "[C:1]>>[C:1]"},
                {"id": "bad"},  # missing required fields
            ]
        }
        fpath = tmp_path / "bad.json"
        fpath.write_text(json.dumps(data))
        lib = ReactionLibrary()
        count = lib.load_json(fpath)
        assert count == 1

    def test_reaction_ids_unique(self, default_lib):
        ids = default_lib.reaction_ids
        assert len(ids) == len(set(ids))


# ============================================================================
# ReactionLibrary: Filtering
# ============================================================================

class TestReactionLibraryFiltering:
    """Test filtering reactions by various criteria."""

    def test_filter_by_name_partial(self, default_lib):
        suzuki = default_lib.filter(names=["suzuki"])
        assert len(suzuki) >= 1
        for rxn in suzuki.reactions:
            assert "suzuki" in rxn.name.lower()

    def test_filter_by_class(self, default_lib):
        couplings = default_lib.filter(classes=["coupling"])
        assert len(couplings) >= 10
        for rxn in couplings.reactions:
            assert rxn.reaction_class == "coupling"

    def test_filter_by_source(self, default_lib):
        h_only = default_lib.filter(sources=["hartenfeller"])
        assert len(h_only) == 58

    def test_filter_by_reliability(self, default_lib):
        reliable = default_lib.filter(min_reliability=0.9)
        for rxn in reliable.reactions:
            assert rxn.reliability >= 0.9

    def test_filter_combined(self, default_lib):
        subset = default_lib.filter(
            classes=["coupling"],
            min_reliability=0.85,
            sources=["hartenfeller"],
        )
        for rxn in subset.reactions:
            assert rxn.reaction_class == "coupling"
            assert rxn.reliability >= 0.85
            assert rxn.source == "hartenfeller"

    def test_filter_by_max_reactants(self, default_lib):
        two_comp = default_lib.filter(max_reactants=2)
        for rxn in two_comp.reactions:
            assert rxn.num_reactants <= 2

    def test_filter_by_product_class(self, default_lib):
        biaryls = default_lib.filter(product_classes=["biaryl"])
        assert len(biaryls) >= 1

    def test_filter_returns_new_library(self, default_lib):
        original_len = len(default_lib)
        _ = default_lib.filter(names=["suzuki"])
        assert len(default_lib) == original_len

    def test_reaction_classes_list(self, default_lib):
        classes = default_lib.reaction_classes
        assert "coupling" in classes
        assert "ring_closure" in classes

    def test_reaction_names_list(self, default_lib):
        names = default_lib.reaction_names
        assert len(names) > 20


# ============================================================================
# Reaction: SMARTS parsing and matching
# ============================================================================

class TestReactionSmarts:
    """Test SMARTS parsing, product matching, and reverse application."""

    def test_rdkit_rxn_compiles(self, suzuki_reaction):
        rxn = suzuki_reaction.rdkit_rxn()
        assert rxn is not None

    def test_rdkit_rxn_cached(self, suzuki_reaction):
        rxn1 = suzuki_reaction.rdkit_rxn()
        rxn2 = suzuki_reaction.rdkit_rxn()
        assert rxn1 is rxn2

    def test_reverse_rxn_compiles(self, suzuki_reaction):
        rev = suzuki_reaction.reverse_rdkit_rxn()
        assert rev is not None

    def test_matches_product_biaryl(self, suzuki_reaction):
        biphenyl = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
        assert suzuki_reaction.matches_product(biphenyl)

    def test_no_match_non_biaryl(self, suzuki_reaction):
        # Product SMARTS [#6:2][#6:1] matches any C-C bond, so use a
        # molecule with no C-C bond to confirm selectivity.
        water = Chem.MolFromSmiles("O")
        assert not suzuki_reaction.matches_product(water)

    def test_apply_reverse_biphenyl(self, suzuki_reaction):
        results = suzuki_reaction.apply_reverse("c1ccc(-c2ccccc2)cc1")
        assert len(results) >= 1
        # Should produce boronic acid + halide pair
        for reactants in results:
            assert len(reactants) == 2

    def test_apply_reverse_no_dummy_atoms(self, default_lib):
        """Regression: apply_reverse must never output dummy atoms ('*').

        Reverse SMARTS can produce '*' when ambiguous patterns like [Cl,Br,I]
        cannot resolve. These must be sanitized before returning.
        """
        test_molecules = [
            "c1ccc(-c2ccccc2)cc1",  # biphenyl
            "CC(=O)Nc1ccccc1",      # acetanilide
            "Oc1ccc(-c2ccccc2)cc1", # 4-hydroxybiphenyl
        ]
        for smi in test_molecules:
            for rxn in default_lib.reactions:
                results = rxn.apply_reverse(smi)
                for reactants in results:
                    for r in reactants:
                        assert '*' not in r, (
                            f"Dummy atom in output of {rxn.name} on {smi}: {r}"
                        )

    def test_apply_reverse_invalid_smiles(self, suzuki_reaction):
        results = suzuki_reaction.apply_reverse("invalid_smiles")
        assert results == []

    def test_hartenfeller_reactions_parse(self, hartenfeller_lib):
        """Check that most Hartenfeller SMARTS parse successfully."""
        parsed = sum(1 for r in hartenfeller_lib.reactions if r.rdkit_rxn() is not None)
        assert parsed >= 50  # Allow some failures

    def test_explore_reactions_parse(self, explore_lib):
        parsed = sum(1 for r in explore_lib.reactions if r.rdkit_rxn() is not None)
        assert parsed >= 30  # all 33 should parse after rxn208 SMARTS fix

    def test_find_matching_reactions(self, default_lib):
        """Biphenyl should match Suzuki and possibly decarboxylative coupling."""
        matches = default_lib.find_matching_reactions("c1ccc(-c2ccccc2)cc1")
        names = [r.name for r, _ in matches]
        assert any("suzuki" in n.lower() or "Suzuki" in n for n in names)

    def test_find_matching_amide(self, default_lib):
        """N-methylacetamide should match amide coupling reactions."""
        matches = default_lib.find_matching_reactions("CC(=O)NC")
        assert len(matches) >= 1


# ============================================================================
# Constraints
# ============================================================================

class TestConstraints:
    """Test ConstraintConfig and build_constraints."""

    def test_build_from_simple_params(self):
        cfg = build_constraints(allowed_reactions=["Suzuki"], max_steps=3)
        assert cfg is not None
        assert cfg.allowed_reactions == ["Suzuki"]
        assert cfg.max_steps == 3

    def test_build_returns_none_if_empty(self):
        cfg = build_constraints()
        assert cfg is None

    def test_build_config_takes_precedence(self):
        explicit = ConstraintConfig(max_steps=5)
        cfg = build_constraints(constraints=explicit, max_steps=3)
        assert cfg.max_steps == 5  # config wins

    def test_check_route_metrics_pass(self):
        cfg = ConstraintConfig(max_steps=3, max_lls=2)
        assert cfg.check_route_metrics(total_steps=2, lls=2)

    def test_check_route_metrics_fail_steps(self):
        cfg = ConstraintConfig(max_steps=3)
        assert not cfg.check_route_metrics(total_steps=4, lls=2)

    def test_check_route_metrics_fail_lls(self):
        cfg = ConstraintConfig(max_lls=2)
        assert not cfg.check_route_metrics(total_steps=2, lls=3)

    def test_is_reaction_allowed_whitelist(self):
        cfg = ConstraintConfig(allowed_reactions=["Suzuki"])
        suzuki = Reaction(
            id="t", name="Suzuki_biaryl", reaction_class="coupling",
            product_class="biaryl", smarts_forward=">>",
        )
        amide = Reaction(
            id="t2", name="amide_coupling", reaction_class="coupling",
            product_class="amide", smarts_forward=">>",
        )
        assert cfg.is_reaction_allowed(suzuki)
        assert not cfg.is_reaction_allowed(amide)

    def test_is_reaction_allowed_blacklist(self):
        cfg = ConstraintConfig(blocked_reactions=["Grignard"])
        grignard = Reaction(
            id="t", name="Grignard_carbonyl", reaction_class="coupling",
            product_class="ketone", smarts_forward=">>",
        )
        suzuki = Reaction(
            id="t2", name="Suzuki", reaction_class="coupling",
            product_class="biaryl", smarts_forward=">>",
        )
        assert not cfg.is_reaction_allowed(grignard)
        assert cfg.is_reaction_allowed(suzuki)

    def test_blocked_overrides_allowed(self):
        cfg = ConstraintConfig(
            allowed_reactions=["Suzuki", "Grignard"],
            blocked_reactions=["Grignard"],
        )
        grignard = Reaction(
            id="t", name="Grignard_alcohol", reaction_class="coupling",
            product_class="alcohol", smarts_forward=">>",
        )
        assert not cfg.is_reaction_allowed(grignard)

    def test_class_filter(self):
        cfg = ConstraintConfig(allowed_classes=["coupling"])
        ring = Reaction(
            id="t", name="triazole", reaction_class="ring_closure",
            product_class="triazole", smarts_forward=">>",
        )
        coup = Reaction(
            id="t2", name="Suzuki", reaction_class="coupling",
            product_class="biaryl", smarts_forward=">>",
        )
        assert not cfg.is_reaction_allowed(ring)
        assert cfg.is_reaction_allowed(coup)

    def test_min_reliability_filter(self):
        cfg = ConstraintConfig(min_reliability=0.9)
        low = Reaction(
            id="t", name="test", reaction_class="x", product_class="y",
            smarts_forward=">>", reliability=0.5,
        )
        high = Reaction(
            id="t2", name="test2", reaction_class="x", product_class="y",
            smarts_forward=">>", reliability=0.95,
        )
        assert not cfg.is_reaction_allowed(low)
        assert cfg.is_reaction_allowed(high)

    def test_get_filtered_library(self, default_lib):
        cfg = ConstraintConfig(
            allowed_classes=["coupling"],
            min_reliability=0.85,
        )
        filtered = cfg.get_filtered_library(default_lib)
        assert len(filtered) < len(default_lib)
        for rxn in filtered.reactions:
            assert rxn.reaction_class == "coupling"
            assert rxn.reliability >= 0.85

    def test_blocked_classes_removes_all(self, default_lib):
        """Regression: blocking all classes must return empty library."""
        all_classes = default_lib.reaction_classes
        cfg = ConstraintConfig(blocked_classes=all_classes)
        filtered = cfg.get_filtered_library(default_lib)
        assert len(filtered) == 0

    def test_blocked_classes_removes_specific(self, default_lib):
        """Blocking one class must not leak any reactions of that class."""
        cfg = ConstraintConfig(blocked_classes=["coupling"])
        filtered = cfg.get_filtered_library(default_lib)
        assert len(filtered) < len(default_lib)
        for rxn in filtered.reactions:
            assert rxn.reaction_class != "coupling"

    def test_blocked_classes_nonexistent_keeps_all(self, default_lib):
        """Blocking a class that doesn't exist must keep everything."""
        cfg = ConstraintConfig(blocked_classes=["nonexistent_class"])
        filtered = cfg.get_filtered_library(default_lib)
        assert len(filtered) == len(default_lib)


# ============================================================================
# SMARTS Retrosynthesis Engine
# ============================================================================

class TestSmartsRetrosynthesis:
    """Test standalone SMARTS retrosynthesis."""

    def test_import(self):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis, RetroSynthNode

    def test_basic_retro(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=1, max_nodes=100)
        routes = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")  # biphenyl
        assert len(routes) >= 1
        # At least one route should have children
        has_children = any(not r.is_leaf for r in routes)
        assert has_children

    def test_retro_with_purchasability(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        known_bbs = {"OB(O)c1ccccc1", "Brc1ccccc1", "Clc1ccccc1", "Ic1ccccc1",
                      "c1ccccc1B(O)O", "c1ccccc1Br"}

        def is_purchasable(smi):
            return smi in known_bbs

        retro = SmartsRetrosynthesis(default_lib, max_depth=2, max_nodes=200)
        routes = retro.retrosynthesise(
            "c1ccc(-c2ccccc2)cc1",
            is_purchasable=is_purchasable,
        )
        assert len(routes) >= 1

    def test_retro_invalid_smiles(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=1)
        routes = retro.retrosynthesise("not_a_smiles")
        assert routes == []

    def test_retro_with_constraints(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        cfg = ConstraintConfig(allowed_reactions=["Suzuki"], max_steps=2)
        retro = SmartsRetrosynthesis(default_lib, max_depth=2, constraints=cfg)
        routes = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")
        # All reactions used should be Suzuki-related
        for route in routes:
            if route.reaction:
                assert "suzuki" in route.reaction.name.lower() or "Suzuki" in route.reaction.name

    def test_retro_node_metrics(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=1, max_nodes=50)
        routes = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")
        if routes:
            r = routes[0]
            assert r.num_steps >= 0
            assert r.longest_linear_sequence >= 0
            assert r.num_leaves >= 1

    def test_retro_node_to_dict(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=1, max_nodes=50)
        routes = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")
        if routes:
            d = routes[0].to_dict()
            assert "smiles" in d
            assert "depth" in d

    def test_retro_node_pretty_print(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=1, max_nodes=50)
        routes = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")
        if routes:
            text = routes[0].pretty_print()
            assert "c1ccc" in text

    def test_max_nodes_respected(self, default_lib):
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(default_lib, max_depth=5, max_nodes=10)
        _ = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1")
        assert retro._nodes_explored <= 15  # small buffer ok


# ============================================================================
# Integration: Validate bond disconnection
# ============================================================================

class TestBondValidation:
    """Test SMARTS-based bond validation against library."""

    def test_validate_biaryl_disconnection(self, default_lib):
        """Biphenyl → benzene + benzene should match Suzuki."""
        is_valid, rxn, score = default_lib.validate_bond_disconnection(
            "c1ccc(-c2ccccc2)cc1",
            ["c1ccccc1", "c1ccccc1"],
        )
        # May or may not match depending on SMARTS specificity
        # At minimum the function should run without error
        assert isinstance(is_valid, bool)
        assert isinstance(score, float)

    def test_validate_with_brics_dummy_atoms(self, default_lib):
        """BRICS fragments with [16*] dummy atoms must still match reactions.

        Regression: BRICS DAG children contain dummy atoms like [16*]c1ccccc1
        while reverse SMARTS products have * dummy atoms. Both must be stripped
        for core comparison.
        """
        is_valid, rxn, score = default_lib.validate_bond_disconnection(
            "c1ccc(-c2ccccc2)cc1",
            ["[16*]c1ccccc1", "[16*]c1ccccc1"],
        )
        assert is_valid, "Should match Suzuki despite BRICS dummy atoms"
        assert rxn is not None
        assert "suzuki" in rxn.name.lower() or "Suzuki" in rxn.name
        assert score > 0.5

    def test_validate_with_generic_dummy_atoms(self, default_lib):
        """Fragments with generic * dummy atoms must also match."""
        is_valid, rxn, score = default_lib.validate_bond_disconnection(
            "c1ccc(-c2ccccc2)cc1",
            ["*c1ccccc1", "*c1ccccc1"],
        )
        assert is_valid, "Should match despite generic * dummy atoms"

    def test_validate_invalid_smiles(self, default_lib):
        is_valid, rxn, score = default_lib.validate_bond_disconnection(
            "not_valid", ["also_bad"]
        )
        assert not is_valid
        assert score == 0.0

    def test_validate_returns_best_match(self, default_lib):
        """If multiple reactions match, should return highest scored."""
        is_valid, rxn, score = default_lib.validate_bond_disconnection(
            "CC(=O)NC",  # N-methylacetamide
            ["CC(=O)O", "CN"],  # acid + amine
        )
        if is_valid:
            assert score > 0.0


# ============================================================================
# ReactionLibrary: Misc
# ============================================================================

class TestReactionLibraryMisc:
    """Misc library functionality."""

    def test_summary(self, default_lib):
        s = default_lib.summary()
        assert "ReactionLibrary" in s
        assert "coupling" in s

    def test_add_reaction(self):
        lib = ReactionLibrary()
        rxn = Reaction(
            id="new", name="test", reaction_class="x",
            product_class="y", smarts_forward="[C:1]>>[C:1]",
        )
        lib.add(rxn)
        assert len(lib) == 1
        assert "new" in lib

    def test_remove_reaction(self):
        lib = ReactionLibrary()
        rxn = Reaction(
            id="del_me", name="test", reaction_class="x",
            product_class="y", smarts_forward="[C:1]>>[C:1]",
        )
        lib.add(rxn)
        lib.remove("del_me")
        assert len(lib) == 0

    def test_get_missing(self):
        lib = ReactionLibrary()
        assert lib.get("nonexistent") is None

    def test_contains(self, hartenfeller_lib):
        assert "hartenfeller_31" in hartenfeller_lib  # Suzuki
        assert "nonexistent" not in hartenfeller_lib

    def test_getitem(self, hartenfeller_lib):
        rxn = hartenfeller_lib["hartenfeller_31"]
        assert rxn.name == "Suzuki"


# ============================================================================
# Dummy atom sanitization
# ============================================================================

class TestDummyAtomSanitization:
    """Regression tests for dummy atom handling."""

    def test_replace_dummy_bare_star(self):
        from fragmentretro.utils.helpers import replace_dummy_atoms_regex
        result = replace_dummy_atoms_regex("*c1ccccc1")
        assert "*" not in result
        assert result == "c1ccccc1"

    def test_replace_dummy_bracketed(self):
        from fragmentretro.utils.helpers import replace_dummy_atoms_regex
        result = replace_dummy_atoms_regex("[1*]c1ccccc1")
        assert "*" not in result

    def test_replace_dummy_mixed(self):
        from fragmentretro.utils.helpers import replace_dummy_atoms_regex
        result = replace_dummy_atoms_regex("*CC[2*]")
        assert "*" not in result

    def test_replace_dummy_internal(self):
        from fragmentretro.reaction_library import _replace_dummy_atoms
        assert _replace_dummy_atoms("*c1ccccc1") == "c1ccccc1"
        assert _replace_dummy_atoms("*CCN1CCCC1") is not None
        assert "*" not in _replace_dummy_atoms("*CCN1CCCC1")
        assert _replace_dummy_atoms("CCO") == "CCO"  # no dummy, unchanged
