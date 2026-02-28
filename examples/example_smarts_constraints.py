#!/usr/bin/env python
"""Example: SMARTS reaction library, constrained scoring, and retrosynthesis.

Demonstrates:
  1. Loading reaction libraries (Hartenfeller, eXplore, custom)
  2. Filtering reactions by name, class, reliability
  3. Tier 2 scoring: BRICS + SMARTS validation
  4. Tier 3 scoring: Standalone SMARTS retrosynthesis
  5. Using ConstraintConfig for complex constraint specs

Usage:
    python example_smarts_constraints.py \\
        --smiles "c1ccc(-c2ccccc2)cc1" \\
        --mol-properties ../mol_properties.json

    python example_smarts_constraints.py \\
        --smiles "CC(=O)Nc1ccc(-c2ccccc2)cc1" \\
        --mol-properties ../mol_properties.json \\
        --allowed-reactions Suzuki amide_coupling \\
        --max-steps 3
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development use
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fragmentretro.reaction_library import ReactionLibrary
from fragmentretro.constraints import ConstraintConfig, build_constraints


def demo_library():
    """Demonstrate loading and browsing the reaction library."""
    print("=" * 70)
    print("1. REACTION LIBRARY LOADING & BROWSING")
    print("=" * 70)

    # Load all built-in reactions
    lib = ReactionLibrary.default()
    print(f"\nDefault library (Hartenfeller + eXplore):")
    print(lib.summary())

    # Filter by class
    couplings = lib.filter(classes=["coupling"])
    print(f"\nCoupling reactions: {len(couplings)}")
    for rxn in sorted(couplings.reactions, key=lambda r: r.name)[:10]:
        print(f"  {rxn.id}: {rxn.name} (reliability={rxn.reliability})")

    # Filter by name (partial, case-insensitive)
    suzuki = lib.filter(names=["suzuki"])
    print(f"\nSuzuki-like reactions: {len(suzuki)}")
    for rxn in suzuki.reactions:
        print(f"  {rxn.id}: {rxn.name} ({rxn.source})")

    # Ring closure reactions
    rings = lib.filter(classes=["ring_closure"])
    print(f"\nRing closure reactions: {len(rings)}")
    for rxn in sorted(rings.reactions, key=lambda r: r.name)[:10]:
        print(f"  {rxn.id}: {rxn.name} -> {rxn.product_class}")

    return lib


def demo_smarts_matching(lib: ReactionLibrary):
    """Demonstrate SMARTS product matching."""
    print("\n" + "=" * 70)
    print("2. SMARTS PRODUCT MATCHING")
    print("=" * 70)

    targets = [
        ("c1ccc(-c2ccccc2)cc1", "Biphenyl"),
        ("CC(=O)NC", "N-methylacetamide"),
        ("c1ccc(-n2ccnn2)cc1", "1-phenylpyrazole"),
    ]

    for smiles, name in targets:
        matches = lib.find_matching_reactions(smiles, top_k=5)
        print(f"\n{name} ({smiles}):")
        if matches:
            for rxn, score in matches:
                print(f"  ✓ {rxn.name} ({rxn.reaction_class}, reliability={score:.2f})")
        else:
            print("  No matching reactions found")


def demo_reverse_application(lib: ReactionLibrary):
    """Demonstrate reverse SMARTS application (retrosynthetic disconnection)."""
    print("\n" + "=" * 70)
    print("3. REVERSE SMARTS APPLICATION")
    print("=" * 70)

    target = "c1ccc(-c2ccccc2)cc1"  # Biphenyl
    print(f"\nTarget: biphenyl ({target})")

    suzuki_rxns = lib.filter(names=["Suzuki"])
    for rxn in suzuki_rxns.reactions:
        results = rxn.apply_reverse(target)
        if results:
            print(f"\n  {rxn.name} ({rxn.id}):")
            for i, reactants in enumerate(results[:3]):
                print(f"    Set {i+1}: {' + '.join(reactants)}")


def demo_constraints():
    """Demonstrate the constraint system."""
    print("\n" + "=" * 70)
    print("4. CONSTRAINT SYSTEM")
    print("=" * 70)

    # Simple params → auto-builds config
    cfg = build_constraints(
        allowed_reactions=["Suzuki", "amide_coupling"],
        max_steps=3,
        max_lls=2,
    )
    print(f"\nSimple constraint: {cfg}")
    print(f"  Route (steps=2, lls=2): {'PASS' if cfg.check_route_metrics(2, 2) else 'FAIL'}")
    print(f"  Route (steps=4, lls=2): {'PASS' if cfg.check_route_metrics(4, 2) else 'FAIL'}")

    # Complex config
    config = ConstraintConfig(
        allowed_reactions=["Suzuki", "amide_coupling", "Sonogashira"],
        blocked_reactions=["Grignard"],
        allowed_classes=["coupling"],
        max_steps=3,
        max_lls=2,
        min_reliability=0.8,
        require_all_validated=True,
    )
    print(f"\nComplex constraint config:")
    print(f"  Allowed reactions: {config.allowed_reactions}")
    print(f"  Blocked reactions: {config.blocked_reactions}")
    print(f"  Max steps: {config.max_steps}, Max LLS: {config.max_lls}")
    print(f"  Min reliability: {config.min_reliability}")
    print(f"  Require all validated: {config.require_all_validated}")

    # Apply to library
    lib = ReactionLibrary.default()
    filtered = config.get_filtered_library(lib)
    print(f"\n  Original library: {len(lib)} reactions")
    print(f"  After constraints: {len(filtered)} reactions")
    for rxn in filtered.reactions:
        print(f"    {rxn.name} (class={rxn.reaction_class}, reliability={rxn.reliability})")


def demo_retrosynthesis(lib: ReactionLibrary):
    """Demonstrate standalone SMARTS retrosynthesis."""
    print("\n" + "=" * 70)
    print("5. STANDALONE SMARTS RETROSYNTHESIS")
    print("=" * 70)

    from fragmentretro.smarts_retro import SmartsRetrosynthesis

    # Simple retrosynthesis
    retro = SmartsRetrosynthesis(lib, max_depth=2, max_nodes=200)
    target = "c1ccc(-c2ccccc2)cc1"  # Biphenyl
    print(f"\nTarget: biphenyl ({target})")

    routes = retro.retrosynthesise(target, max_routes=3)
    print(f"Found {len(routes)} routes")

    for i, route in enumerate(routes[:3]):
        print(f"\n  Route {i+1}:")
        print(f"  {route.pretty_print()}")

    # Retrosynthesis with constraints
    print("\n--- With constraints (Suzuki only, max 2 steps) ---")
    cfg = ConstraintConfig(allowed_reactions=["Suzuki"], max_steps=2)
    retro_constrained = SmartsRetrosynthesis(lib, max_depth=2, max_nodes=100, constraints=cfg)
    routes_c = retro_constrained.retrosynthesise(target, max_routes=3)
    print(f"Found {len(routes_c)} constrained routes")
    for i, route in enumerate(routes_c[:2]):
        print(f"\n  Route {i+1}:")
        print(f"  {route.pretty_print()}")


def demo_tiered_scoring(smiles: str, mol_properties_path: str,
                        allowed_reactions=None, max_steps=None):
    """Demonstrate Tier 1/2/3 scoring."""
    print("\n" + "=" * 70)
    print("6. TIERED SCORING")
    print("=" * 70)

    from fragmentretro.scoring import (
        compute_score,
        compute_score_with_dag,
        compute_score_validated,
        compute_score_smarts,
    )
    from fragmentretro.retrosynthesis import CompoundFilter

    cf = CompoundFilter(mol_properties_path)
    lib = ReactionLibrary.default()

    print(f"\nTarget: {smiles}")
    if allowed_reactions:
        print(f"Allowed reactions: {allowed_reactions}")
    if max_steps:
        print(f"Max steps: {max_steps}")

    # Tier 1: BRICS-only
    t1_score = compute_score(smiles, cf)
    print(f"\n  Tier 1 (BRICS-only):         {t1_score:.4f}")

    # Tier 1b: BRICS with DAG
    t1b = compute_score_with_dag(smiles, cf)
    print(f"  Tier 1b (BRICS + DAG):       {t1b.score:.4f}  "
          f"(steps={t1b.total_steps}, LLS={t1b.longest_linear_sequence}, "
          f"BBs={t1b.num_building_blocks})")

    # Tier 2: BRICS + SMARTS validation
    t2 = compute_score_validated(
        smiles, cf, lib,
        allowed_reactions=allowed_reactions,
        max_steps=max_steps,
    )
    print(f"  Tier 2 (BRICS + SMARTS):     {t2.score:.4f}  "
          f"(validation={t2.validation_coverage:.0%}, "
          f"matched={len(t2.matched_reactions)} rxns, "
          f"constraints={'✓' if t2.constraints_satisfied else '✗'})")
    if t2.matched_reactions:
        for rxn_name, score in t2.matched_reactions:
            print(f"    ✓ {rxn_name} (score={score:.2f})")

    # Tier 3: SMARTS-only
    t3 = compute_score_smarts(
        smiles, cf, lib,
        allowed_reactions=allowed_reactions,
        max_steps=max_steps,
        max_depth=2,
        max_nodes=200,
    )
    print(f"  Tier 3 (SMARTS-only):        {t3:.4f}")


def demo_custom_library():
    """Demonstrate loading a custom user reaction file."""
    print("\n" + "=" * 70)
    print("7. CUSTOM REACTION LIBRARY")
    print("=" * 70)

    import tempfile

    custom_data = {
        "reactions": [
            {
                "id": "my_reaction_001",
                "name": "my_special_coupling",
                "class": "coupling",
                "product_class": "biaryl",
                "smarts_forward": "[c:1]B(O)O.[c:2][Br]>>[c:1][c:2]",
                "num_reactants": 2,
                "reliability": 0.95,
                "notes": "My lab's optimized Suzuki variant",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(custom_data, f)
        custom_path = f.name

    # Load default + custom
    lib = ReactionLibrary.default()
    before = len(lib)
    lib.load_json(custom_path, source="my_lab")
    print(f"\n  Before custom: {before} reactions")
    print(f"  After custom:  {len(lib)} reactions")
    custom = lib.filter(sources=["my_lab"])
    print(f"  Custom reactions: {len(custom)}")
    for rxn in custom.reactions:
        print(f"    {rxn.id}: {rxn.name} (reliability={rxn.reliability})")


def main():
    parser = argparse.ArgumentParser(
        description="SMARTS reaction library & constrained scoring demo"
    )
    parser.add_argument("--smiles", type=str, default="c1ccc(-c2ccccc2)cc1",
                        help="Target SMILES (default: biphenyl)")
    parser.add_argument("--mol-properties", type=str, default=None,
                        help="Path to mol_properties.json for scoring demos")
    parser.add_argument("--allowed-reactions", nargs="+", default=None,
                        help="Allowed reaction names for constrained scoring")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum synthesis steps")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip Tier 2/3 scoring (requires mol_properties)")
    args = parser.parse_args()

    # Library demos (no dependencies)
    lib = demo_library()
    demo_smarts_matching(lib)
    demo_reverse_application(lib)
    demo_constraints()
    demo_retrosynthesis(lib)
    demo_custom_library()

    # Scoring demos (require mol_properties)
    if not args.skip_scoring and args.mol_properties:
        demo_tiered_scoring(
            args.smiles,
            args.mol_properties,
            allowed_reactions=args.allowed_reactions,
            max_steps=args.max_steps,
        )
    elif not args.skip_scoring:
        print("\n(Skipping scoring demos — provide --mol-properties to enable)")


if __name__ == "__main__":
    main()
