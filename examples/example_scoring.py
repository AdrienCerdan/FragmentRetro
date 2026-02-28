#!/usr/bin/env python
"""Example: compare FragmentRetro scoring methods on a target molecule.

Usage:
    python example_scoring.py

Requirements:
    pip install -e ".[dev]"
"""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

from fragmentretro.scoring import (
    compute_score,
    compute_score_with_dag,
    is_feasible,
    load_compound_filter,
)
from fragmentretro.utils.filter_compound import precompute_properties


# ---------------------------------------------------------------------------
# 1. Setup: create a small BB stock and precompute properties
# ---------------------------------------------------------------------------

BUILDING_BLOCKS = [
    # amines
    "Nc1ccccc1", "NCC", "NCCN", "NC(=O)c1ccccc1", "Nc1ccc(O)cc1",
    "Nc1ccc(Cl)cc1", "Nc1ccc(F)cc1", "Nc1ccc(C)cc1", "NCC(=O)O",
    "N1CCNCC1", "NC1CCCCC1", "Nc1ccncc1", "Nc1ccc(OC)cc1",
    # acids / acyl
    "CC(=O)O", "OC(=O)c1ccccc1", "OC(=O)CCl", "CC(=O)Cl",
    "OC(=O)c1ccc(Cl)cc1", "OC(=O)c1ccc(F)cc1",
    # alcohols / phenols
    "Oc1ccccc1", "OCC", "OCCO", "Oc1ccc(C)cc1", "OC",
    # aryl halides (for cross-coupling)
    "Clc1ccccc1", "Brc1ccccc1", "Brc1ccc(C)cc1", "Brc1ccc(F)cc1",
    "Clc1ccc(Cl)cc1", "Brc1ccncc1",
    # heterocycles
    "c1ccncc1", "c1cc[nH]c1", "c1ccc2[nH]ccc2c1",
    # fragments / simple chains
    "CCCC", "CCC", "CC", "CCO", "CCN", "CCCN", "CCCCN",
    "CCCNC", "NCCCN", "c1ccccc1", "c1ccc(N)cc1",
    # sulfonamides
    "CS(=O)(=O)Cl", "c1ccc(S(=O)(=O)Cl)cc1",
    # boronic acids (Suzuki)
    "OB(O)c1ccccc1", "OB(O)c1ccc(C)cc1",
]

# Target molecules to test (increasing complexity)
TARGETS = [
    ("CCN",                                     "Ethylamine (trivial BB)"),
    ("CC(=O)Nc1ccccc1",                        "Acetanilide (2 fragments)"),
    ("CC(=O)Nc1ccc(OC)cc1",                    "p-Methoxyacetanilide (3 fragments)"),
    ("CC(=O)Nc1ccc(-c2ccccc2)cc1",             "4-Biphenylacetamide (3-4 fragments)"),
    ("O=C(Nc1ccc(S(=O)(=O)N2CCNCC2)cc1)c1ccccc1", "Sulfonamide-amide (complex)"),
    ("C1CC2CCCC3CCCC1C23",                      "Polycyclic (likely infeasible)"),
]


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        props_path = Path(tmpdir) / "bb_properties.json"

        print("=" * 72)
        print("Precomputing BB properties...")
        t0 = time.perf_counter()
        precompute_properties(BUILDING_BLOCKS, props_path)
        print(f"  {len(BUILDING_BLOCKS)} BBs precomputed in {time.perf_counter() - t0:.2f}s")

        print("\nLoading CompoundFilter...")
        cf = load_compound_filter(props_path)
        print("  Ready.\n")

        # -------------------------------------------------------------------
        # 2. Run all three scoring methods on each target
        # -------------------------------------------------------------------
        for smiles, name in TARGETS:
            print("=" * 72)
            print(f"TARGET: {name}")
            print(f"SMILES: {smiles}\n")

            # --- Binary ---
            t0 = time.perf_counter()
            feasible = is_feasible(smiles, cf)
            dt_binary = time.perf_counter() - t0
            print(f"[1] is_feasible():  {feasible}  ({dt_binary*1000:.1f} ms)")

            # --- Continuous (no DAG) ---
            t0 = time.perf_counter()
            score = compute_score(smiles, cf)
            dt_cont = time.perf_counter() - t0
            print(f"[2] compute_score(): {score:.3f}  ({dt_cont*1000:.1f} ms)")

            # --- DAG scoring ---
            t0 = time.perf_counter()
            result = compute_score_with_dag(smiles, cf)
            dt_dag = time.perf_counter() - t0
            print(f"[3] compute_score_with_dag(): ({dt_dag*1000:.1f} ms)")
            print(f"      score:         {result.score:.3f}")
            print(f"      feasible:      {result.feasible}")
            print(f"      total_steps:   {result.total_steps}")
            print(f"      LLS:           {result.longest_linear_sequence}")
            print(f"      num_BBs:       {result.num_building_blocks}")
            print(f"      convergence:   {result.convergence_score:.2f}")

            # Print DAG tree if available
            if result.dag is not None:
                print(f"\n  Route tree:")
                print("  " + result.dag.pretty_print().replace("\n", "\n  "))
            print()


if __name__ == "__main__":
    main()
