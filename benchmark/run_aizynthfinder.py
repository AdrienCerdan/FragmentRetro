#!/usr/bin/env python
"""Benchmark AiZynthFinder on the same target set.

Prerequisites:
    pip install aizynthfinder[all]
    download_public_data aizynthfinder

Usage:
    python run_aizynthfinder.py \
        --config ./benchmark_data/benchmark_config.json \
        --time-limit 120

    # With custom model
    python run_aizynthfinder.py \
        --config ./benchmark_data/benchmark_config.json \
        --policy-model /path/to/model.onnx \
        --template-file /path/to/templates.hdf5

    # With YAML config
    python run_aizynthfinder.py \
        --config ./benchmark_data/benchmark_config.json \
        --aizynthfinder-config /path/to/aizynthfinder_config.yml
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from rdkit import Chem


def find_model_files(args):
    """Auto-detect AiZynthFinder model files."""
    if args.policy_model and args.template_file:
        return Path(args.policy_model), Path(args.template_file)

    home = Path.home()
    search_dirs = [
        home / ".aizynthfinder",
        home / "aizynthfinder_data",
        Path("/opt/aizynthfinder"),
        Path("./aizynthfinder_data"),
    ]

    model_path = Path(args.policy_model) if args.policy_model else None
    template_path = Path(args.template_file) if args.template_file else None

    for d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix == ".onnx" and not model_path:
                model_path = f
            if f.suffix == ".hdf5" and "template" in f.name.lower() and not template_path:
                template_path = f

    return model_path, template_path


def run_single(finder, smiles, time_limit):
    """Run AiZynthFinder on a single molecule."""
    t0 = time.time()
    try:
        finder.target_smiles = smiles
        finder.config.search.time_limit = time_limit
        finder.config.search.iteration_limit = 500

        finder.tree_search()
        finder.build_routes()
        search_time = time.time() - t0

        stats = finder.routes.compute_scores(*finder.scorers.objects())
        n_routes = len(finder.routes)
        solved = False
        best_route = None

        if n_routes > 0:
            routes_list = list(finder.routes)
            solved = any(r.is_solved for r in routes_list)

            solved_routes = [r for r in routes_list if r.is_solved]
            top = solved_routes[0] if solved_routes else routes_list[0]

            best_route = {
                "solved": top.is_solved,
                "n_steps": len(list(top.reactions)),
                "n_bbs": len(list(top.leaves)),
            }

            try:
                route_scores = stats.get(top, {})
                if isinstance(route_scores, dict):
                    best_route["scores"] = {
                        k: round(float(v), 4) for k, v in route_scores.items()
                    }
            except Exception:
                pass

        return {
            "solved": solved,
            "n_routes": n_routes,
            "n_solved_routes": sum(1 for r in finder.routes if r.is_solved) if n_routes > 0 else 0,
            "time_s": round(search_time, 4),
            "best_route": best_route,
        }

    except Exception as e:
        return {
            "solved": False, "n_routes": 0, "n_solved_routes": 0,
            "time_s": round(time.time() - t0, 4),
            "best_route": None,
            "error": str(e),
        }


def compute_summary(results):
    valid = [r for r in results if r["valid"]]
    n = len(valid)

    solved = sum(1 for r in valid if r["aizynthfinder"]["solved"])
    times = [r["aizynthfinder"]["time_s"] for r in valid]

    summary = {
        "n_targets": len(results),
        "n_valid": n,
        "solved": solved,
        "solve_rate": round(solved / n, 4) if n else 0,
        "mean_time_s": round(sum(times) / n, 4) if n else 0,
        "median_time_s": round(sorted(times)[n//2], 4) if n else 0,
        "total_time_s": round(sum(times), 2),
        "min_time_s": round(min(times), 4) if times else 0,
        "max_time_s": round(max(times), 4) if times else 0,
    }

    solved_results = [r for r in valid if r["aizynthfinder"]["solved"]
                      and r["aizynthfinder"]["best_route"]]
    if solved_results:
        steps = [r["aizynthfinder"]["best_route"]["n_steps"] for r in solved_results]
        bbs = [r["aizynthfinder"]["best_route"]["n_bbs"] for r in solved_results]
        summary["mean_steps"] = round(sum(steps) / len(steps), 2)
        summary["mean_bbs"] = round(sum(bbs) / len(bbs), 2)

    errors = sum(1 for r in valid if "error" in r["aizynthfinder"])
    if errors:
        summary["errors"] = errors

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark AiZynthFinder")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to benchmark_config.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--time-limit", type=float, default=120.0,
                        help="Search time limit per molecule (seconds)")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit to first N targets")
    parser.add_argument("--policy-model", type=str, default=None,
                        help="Path to expansion policy model (.onnx)")
    parser.add_argument("--template-file", type=str, default=None,
                        help="Path to reaction templates (.hdf5)")
    parser.add_argument("--aizynthfinder-config", type=str, default=None,
                        help="Path to AiZynthFinder YAML config")
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

    print(f"AiZynthFinder Benchmark")
    print(f"  Targets:    {len(targets)}")
    print(f"  Stock BBs:  {config['n_stock_bbs']}")
    print(f"  Time limit: {args.time_limit}s/mol")

    # --- Setup AiZynthFinder ---
    from aizynthfinder.aizynthfinder import AiZynthFinder

    stock_file = config.get("aizynthfinder_stock_hdf5",
                            config.get("aizynthfinder_stock"))

    if args.aizynthfinder_config:
        print(f"  Config: {args.aizynthfinder_config}")
        finder = AiZynthFinder(configfile=args.aizynthfinder_config)
    else:
        model_path, template_path = find_model_files(args)
        if not model_path or not template_path:
            print("\nERROR: Could not find AiZynthFinder model files.")
            print("  1. Run: download_public_data aizynthfinder")
            print("  2. Or provide --policy-model and --template-file")
            print("  3. Or provide --aizynthfinder-config")
            sys.exit(1)

        print(f"  Policy model:  {model_path}")
        print(f"  Templates:     {template_path}")
        print(f"  Stock:         {stock_file}")

        config_dict = {
            "search": {
                "time_limit": args.time_limit,
                "iteration_limit": 500,
                "return_first": False,
            },
            "policy": {
                "files": {
                    "expansion": {
                        "uspto": [str(model_path), str(template_path)]
                    }
                }
            },
            "stock": {
                "files": {
                    "stock": str(stock_file)
                }
            },
        }
        finder = AiZynthFinder(configdict=config_dict)

    print(f"\n  AiZynthFinder ready")
    print()

    # --- Run ---
    results = []
    n_total = len(targets)
    t_start = time.time()

    for idx, smiles in enumerate(targets):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [{idx+1}/{n_total}] SKIP (invalid): {smiles[:60]}")
            results.append({"idx": idx, "smiles": smiles, "valid": False,
                            "aizynthfinder": {}})
            continue

        canonical = Chem.MolToSmiles(mol)
        r = run_single(finder, canonical, args.time_limit)

        status = "Y" if r["solved"] else "N"
        routes_str = f"routes={r['n_routes']}"
        if r["best_route"]:
            routes_str += f" steps={r['best_route']['n_steps']}"

        print(f"  [{idx+1}/{n_total}] {canonical[:50]:50s}  "
              f"{status} {routes_str}  ({r['time_s']:.2f}s)")

        results.append({
            "idx": idx, "smiles": canonical,
            "original_smiles": smiles, "valid": True,
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "aizynthfinder": r,
        })

    total_time = time.time() - t_start

    summary = compute_summary(results)
    summary["total_benchmark_time_s"] = round(total_time, 2)
    summary["tool"] = "AiZynthFinder"
    summary["time_limit_per_mol"] = args.time_limit
    summary["config"] = config

    output_path = args.output or str(
        Path(config["targets_file"]).parent / "results_aizynthfinder.json"
    )
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AIZYNTHFINDER BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Targets: {summary['n_valid']}/{summary['n_targets']} valid")
    print(f"Solved:  {summary['solved']}/{summary['n_valid']} ({summary['solve_rate']:.1%})")
    print(f"Time:    {summary['mean_time_s']:.2f}s/mol (total {summary['total_time_s']:.1f}s)")
    if "mean_steps" in summary:
        print(f"Routes:  {summary['mean_steps']:.1f} avg steps, "
              f"{summary['mean_bbs']:.1f} avg BBs")
    if "errors" in summary:
        print(f"Errors:  {summary['errors']}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
