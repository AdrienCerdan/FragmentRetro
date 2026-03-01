#!/usr/bin/env python
"""Benchmark FragmentRetro across all tiers.

Runs the full target set through:
  - Tier 1:  BRICS-only binary / continuous / DAG
  - Tier 2:  BRICS + SMARTS validation
  - Tier 3:  Standalone SMARTS retrosynthesis

Usage:
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json \
        --tiers 1 2 3 \
        --timeout 120

    # Quick test
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json \
        --max-targets 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from fragmentretro.scoring import (
    compute_score,
    compute_score_with_dag,
    compute_score_validated,
    compute_score_smarts,
    is_feasible,
    load_compound_filter,
)
from fragmentretro.reaction_library import ReactionLibrary
from fragmentretro.utils.logging_config import logger


# ---------------------------------------------------------------------------
# Per-tier runners
# ---------------------------------------------------------------------------

def run_tier1_binary(smiles, cf, timeout):
    t0 = time.time()
    try:
        feasible = is_feasible(smiles, cf)
        return {"tier": "1_binary", "solved": feasible,
                "score": 1.0 if feasible else 0.0,
                "time_s": round(time.time() - t0, 4)}
    except Exception as e:
        return {"tier": "1_binary", "solved": False, "score": 0.0,
                "time_s": round(time.time() - t0, 4), "error": str(e)}


def run_tier1_continuous(smiles, cf, timeout):
    t0 = time.time()
    try:
        score = compute_score(smiles, cf)
        return {"tier": "1_continuous", "solved": score > 0,
                "score": round(score, 6),
                "time_s": round(time.time() - t0, 4)}
    except Exception as e:
        return {"tier": "1_continuous", "solved": False, "score": 0.0,
                "time_s": round(time.time() - t0, 4), "error": str(e)}


def run_tier1_dag(smiles, cf, timeout):
    t0 = time.time()
    try:
        r = compute_score_with_dag(smiles, cf)
        return {
            "tier": "1_dag", "solved": r.feasible,
            "score": round(r.score, 6),
            "total_steps": r.total_steps,
            "lls": r.longest_linear_sequence,
            "num_bbs": r.num_building_blocks,
            "convergence": round(r.convergence_score, 4),
            "step_score": round(r.step_score, 4),
            "availability_score": round(r.availability_score, 4),
            "bond_feasibility_score": round(r.bond_feasibility_score, 4),
            "time_s": round(time.time() - t0, 4),
        }
    except Exception as e:
        return {"tier": "1_dag", "solved": False, "score": 0.0,
                "total_steps": 0, "lls": 0, "num_bbs": 0, "convergence": 0.0,
                "time_s": round(time.time() - t0, 4), "error": str(e)}


def run_tier2(smiles, cf, lib, timeout):
    t0 = time.time()
    try:
        r = compute_score_validated(smiles, cf, lib)
        return {
            "tier": "2_validated", "solved": r.feasible,
            "score": round(r.score, 6),
            "total_steps": r.total_steps,
            "lls": r.longest_linear_sequence,
            "num_bbs": r.num_building_blocks,
            "convergence": round(r.convergence_score, 4),
            "validation_coverage": round(r.validation_coverage, 4),
            "matched_reactions": [
                {"name": name, "score": round(s, 4)}
                for name, s in r.matched_reactions
            ],
            "n_matched": len(r.matched_reactions),
            "constraints_satisfied": r.constraints_satisfied,
            "time_s": round(time.time() - t0, 4),
        }
    except Exception as e:
        return {"tier": "2_validated", "solved": False, "score": 0.0,
                "total_steps": 0, "lls": 0, "num_bbs": 0,
                "validation_coverage": 0.0, "n_matched": 0,
                "time_s": round(time.time() - t0, 4), "error": str(e)}


def run_tier3(smiles, cf, lib, timeout, max_depth=3, max_nodes=500):
    t0 = time.time()
    try:
        score = compute_score_smarts(smiles, cf, lib,
                                     max_depth=max_depth, max_nodes=max_nodes)

        routes_info = []
        if score > 0:
            from fragmentretro.smarts_retro import SmartsRetrosynthesis
            retro = SmartsRetrosynthesis(lib, max_depth=max_depth, max_nodes=max_nodes)

            def is_purchasable(smi):
                try:
                    return cf.has_match(smi)
                except Exception:
                    return False

            routes = retro.retrosynthesise(smiles, is_purchasable=is_purchasable,
                                           max_routes=3)
            for r in routes:
                routes_info.append({
                    "solved": r.is_solved,
                    "steps": r.num_steps,
                    "lls": r.longest_linear_sequence,
                    "n_bbs": r.num_leaves,
                    "avg_reliability": round(r.avg_reliability, 4),
                    "reaction": r.reaction.name if r.reaction else None,
                })

        best_route_solved = (routes_info[0]["solved"] if routes_info else False)

        return {
            "tier": "3_smarts", "solved": best_route_solved,
            "score": round(score, 6),
            "n_routes": len(routes_info),
            "best_route": routes_info[0] if routes_info else None,
            "time_s": round(time.time() - t0, 4),
        }
    except Exception as e:
        return {"tier": "3_smarts", "solved": False, "score": 0.0,
                "n_routes": 0, "best_route": None,
                "time_s": round(time.time() - t0, 4), "error": str(e)}


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(targets, config, tiers, timeout=120.0,
                  max_depth=3, max_nodes=500):
    mol_props_path = config["fragmentretro_stock"]
    print(f"Loading CompoundFilter from {mol_props_path} ...")
    cf = load_compound_filter(mol_props_path)
    print(f"  {cf.len_BBs} building blocks loaded")

    lib = None
    if any(t in tiers for t in ["2", "3"]):
        print("Loading ReactionLibrary (91 reactions) ...")
        lib = ReactionLibrary.default()
        print(f"  {len(lib)} reactions loaded")

    results = []
    n_total = len(targets)

    for idx, smiles in enumerate(targets):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [{idx+1}/{n_total}] SKIP (invalid): {smiles[:60]}")
            results.append({"idx": idx, "smiles": smiles, "valid": False, "tiers": {}})
            continue

        canonical = Chem.MolToSmiles(mol)
        mol_result = {
            "idx": idx,
            "smiles": canonical,
            "original_smiles": smiles,
            "valid": True,
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "tiers": {},
        }

        tier_labels = []

        if "1" in tiers or "1b" in tiers:
            r = run_tier1_binary(canonical, cf, timeout)
            mol_result["tiers"]["1_binary"] = r
            tier_labels.append(f"T1b={'Y' if r['solved'] else 'N'}")

        if "1" in tiers or "1c" in tiers:
            r = run_tier1_continuous(canonical, cf, timeout)
            mol_result["tiers"]["1_continuous"] = r
            tier_labels.append(f"T1c={r['score']:.3f}")

        if "1" in tiers or "1d" in tiers:
            r = run_tier1_dag(canonical, cf, timeout)
            mol_result["tiers"]["1_dag"] = r
            tier_labels.append(f"T1d={r['score']:.3f}")

        if "2" in tiers:
            r = run_tier2(canonical, cf, lib, timeout)
            mol_result["tiers"]["2_validated"] = r
            tier_labels.append(f"T2={r['score']:.3f}")

        if "3" in tiers:
            r = run_tier3(canonical, cf, lib, timeout, max_depth, max_nodes)
            mol_result["tiers"]["3_smarts"] = r
            tier_labels.append(f"T3={r['score']:.3f}")

        total_time = sum(t.get("time_s", 0) for t in mol_result["tiers"].values())
        status = " | ".join(tier_labels)
        print(f"  [{idx+1}/{n_total}] {canonical[:50]:50s}  {status}  ({total_time:.2f}s)")

        results.append(mol_result)

    return results


def compute_summary(results):
    valid = [r for r in results if r["valid"]]
    n = len(valid)
    summary = {"n_targets": len(results), "n_valid": n}

    tier_names = set()
    for r in valid:
        tier_names.update(r["tiers"].keys())

    for tier in sorted(tier_names):
        tier_results = [r["tiers"][tier] for r in valid if tier in r["tiers"]]
        if not tier_results:
            continue

        solved = sum(1 for t in tier_results if t.get("solved", False))
        scores = [t["score"] for t in tier_results]
        times = [t["time_s"] for t in tier_results]

        stats = {
            "n_molecules": len(tier_results),
            "solved": solved,
            "solve_rate": round(solved / len(tier_results), 4) if tier_results else 0,
            "mean_score": round(sum(scores) / len(scores), 4),
            "median_score": round(sorted(scores)[len(scores)//2], 4),
            "mean_time_s": round(sum(times) / len(times), 4),
            "median_time_s": round(sorted(times)[len(times)//2], 4),
            "total_time_s": round(sum(times), 2),
            "min_time_s": round(min(times), 4),
            "max_time_s": round(max(times), 4),
        }

        steps = [t["total_steps"] for t in tier_results if "total_steps" in t and t.get("solved")]
        lls_vals = [t["lls"] for t in tier_results if "lls" in t and t.get("solved")]
        bbs = [t["num_bbs"] for t in tier_results if "num_bbs" in t and t.get("solved")]

        if steps:
            stats["mean_steps"] = round(sum(steps) / len(steps), 2)
        if lls_vals:
            stats["mean_lls"] = round(sum(lls_vals) / len(lls_vals), 2)
        if bbs:
            stats["mean_bbs"] = round(sum(bbs) / len(bbs), 2)

        coverages = [t["validation_coverage"] for t in tier_results
                     if "validation_coverage" in t and t.get("solved")]
        if coverages:
            stats["mean_validation_coverage"] = round(sum(coverages) / len(coverages), 4)

        errors = sum(1 for t in tier_results if "error" in t)
        if errors:
            stats["errors"] = errors

        summary[tier] = stats

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark FragmentRetro")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to benchmark_config.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--tiers", type=str, nargs="+", default=["1", "2", "3"],
                        help="Tiers to run: 1 1b 1c 1d 2 3 (default: 1 2 3)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Timeout per molecule (seconds)")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit to first N targets")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Tier 3 max tree depth")
    parser.add_argument("--max-nodes", type=int, default=500,
                        help="Tier 3 max nodes to explore")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    targets = []
    with open(config["targets_file"]) as f:
        for line in f:
            smi = line.strip()
            if smi:
                targets.append(smi)

    if args.max_targets:
        targets = targets[:args.max_targets]

    print(f"FragmentRetro Benchmark")
    print(f"  Targets:    {len(targets)}")
    print(f"  Stock BBs:  {config['n_stock_bbs']}")
    print(f"  Tiers:      {args.tiers}")
    print(f"  Timeout:    {args.timeout}s/mol")
    print()

    t_start = time.time()
    results = run_benchmark(targets, config, args.tiers,
                            timeout=args.timeout,
                            max_depth=args.max_depth,
                            max_nodes=args.max_nodes)
    total_time = time.time() - t_start

    summary = compute_summary(results)
    summary["total_benchmark_time_s"] = round(total_time, 2)
    summary["tool"] = "FragmentRetro"
    summary["tiers_run"] = args.tiers
    summary["config"] = config

    output_path = args.output or str(
        Path(config["targets_file"]).parent / "results_fragmentretro.json"
    )
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FRAGMENTRETRO BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Targets: {summary['n_valid']}/{summary['n_targets']} valid")
    print(f"Total time: {total_time:.1f}s")
    print()

    for tier in sorted(k for k in summary if k.startswith(("1_", "2_", "3_"))):
        s = summary[tier]
        print(f"  {tier:20s}  solve={s['solve_rate']:.1%}  "
              f"score={s['mean_score']:.3f}  "
              f"time={s['mean_time_s']:.3f}s/mol  "
              f"(total={s['total_time_s']:.1f}s)")
        if "mean_steps" in s:
            print(f"  {'':20s}  steps={s['mean_steps']:.1f}  "
                  f"lls={s.get('mean_lls', 'N/A')}  "
                  f"bbs={s.get('mean_bbs', 'N/A')}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
