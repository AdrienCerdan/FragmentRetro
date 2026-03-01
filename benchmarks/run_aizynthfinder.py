#!/usr/bin/env python
"""Benchmark AiZynthFinder on USPTO 190 targets.

Runs AiZynthFinder's MCTS retrosynthesis on the same targets and stock
used for the FragmentRetro benchmark. Results are saved in the same JSON
schema for direct comparison.

Prerequisites:
    pip install aizynthfinder

    AiZynthFinder requires a trained expansion model. You can either:
    1. Download the public USPTO model:
       aizynthcli --config config.yml download-public-data

    2. Or specify your own model files:
       --expansion-model /path/to/model.onnx
       --template-file /path/to/templates.hdf5

Usage:
    # Using a config YAML (recommended)
    python run_aizynthfinder.py \
        --datadir benchmarks/data \
        --config aizynthfinder_config.yml

    # Using explicit model paths
    python run_aizynthfinder.py \
        --datadir benchmarks/data \
        --expansion-model /path/to/model.onnx \
        --template-file /path/to/templates.hdf5

    # Limit to first N targets (quick test)
    python run_aizynthfinder.py \
        --datadir benchmarks/data \
        --config aizynthfinder_config.yml \
        --limit 10

AiZynthFinder config YAML example:
    expansion:
      uspto:
        type: template_based
        model: /path/to/uspto_model.onnx
        template: /path/to/templates.hdf5
    stock:
      emolecules:
        type: inchikey
        path: /path/to/stock.hdf5  # or stock.smi
    search:
      algorithm: mcts
      time_limit: 120
      iteration_limit: 100
      return_first: false
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem


@dataclass
class AiZynthResult:
    """Benchmark result for a single molecule from AiZynthFinder."""
    smiles: str

    # Timing
    wall_time_s: float = 0.0
    timed_out: bool = False
    error: str | None = None

    # Feasibility
    solved: bool = False
    num_routes_found: int = 0

    # Best route quality
    score: float = 0.0
    total_steps: int = 0
    longest_linear_sequence: int = 0
    num_building_blocks: int = 0

    # All routes summary
    all_route_scores: list[float] = field(default_factory=list)

    # Search stats
    num_iterations: int = 0
    num_nodes_explored: int = 0

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "wall_time_s": round(self.wall_time_s, 4),
            "timed_out": self.timed_out,
            "error": self.error,
            "solved": self.solved,
            "num_routes_found": self.num_routes_found,
            "score": round(self.score, 6),
            "total_steps": self.total_steps,
            "longest_linear_sequence": self.longest_linear_sequence,
            "num_building_blocks": self.num_building_blocks,
            "all_route_scores": [round(s, 4) for s in self.all_route_scores],
            "num_iterations": self.num_iterations,
            "num_nodes_explored": self.num_nodes_explored,
        }


def compute_route_lls(tree_dict: dict) -> int:
    """Compute longest linear sequence from an AiZynthFinder route dict."""
    if not tree_dict.get("children"):
        return 0
    child_lls = [compute_route_lls(c) for c in tree_dict["children"]]
    return 1 + max(child_lls) if child_lls else 1


def count_route_leaves(tree_dict: dict) -> int:
    """Count leaf nodes (BBs) in a route dict."""
    if not tree_dict.get("children"):
        return 1
    return sum(count_route_leaves(c) for c in tree_dict["children"])


def count_route_steps(tree_dict: dict) -> int:
    """Count total reaction steps in a route."""
    if not tree_dict.get("children"):
        return 0
    return 1 + sum(count_route_steps(c) for c in tree_dict["children"])


def create_config_dict(
    stock_path: Path,
    expansion_model: Path | None = None,
    template_file: Path | None = None,
    time_limit: int = 120,
    iteration_limit: int = 100,
    max_transforms: int = 25,
    c_exploration: float = 1.4,
) -> dict:
    """Build an AiZynthFinder configuration dictionary."""
    config: dict[str, Any] = {
        "search": {
            "algorithm": "mcts",
            "time_limit": time_limit,
            "iteration_limit": iteration_limit,
            "return_first": False,
        },
        "post_processing": {
            "min_routes": 1,
            "max_routes": 25,
        },
    }

    # Stock
    stock_str = str(stock_path)
    if stock_str.endswith(".hdf5") or stock_str.endswith(".h5"):
        config["stock"] = {"benchmark": {"type": "inchikey", "path": stock_str}}
    else:
        config["stock"] = {"benchmark": {"type": "smi", "path": stock_str}}

    # Expansion model
    if expansion_model and template_file:
        config["expansion"] = {
            "benchmark": {
                "type": "template_based",
                "model": str(expansion_model),
                "template": str(template_file),
            }
        }
    # If neither provided, user must supply --config

    return config


def run_single_aizynthfinder(
    finder,
    smiles: str,
) -> AiZynthResult:
    """Run AiZynthFinder on a single molecule."""
    result = AiZynthResult(smiles=smiles)

    try:
        t0 = time.perf_counter()
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        result.wall_time_s = time.perf_counter() - t0

        # Extract results
        stats = finder.stats
        result.num_iterations = stats.get("iterations", 0)
        result.num_nodes_explored = stats.get("nodes", 0)

        routes = finder.routes
        if hasattr(routes, "route_costs"):
            route_scores = routes.route_costs
        elif hasattr(routes, "scores"):
            route_scores = routes.scores
        else:
            route_scores = []

        result.num_routes_found = len(route_scores) if route_scores else 0

        if result.num_routes_found > 0:
            result.all_route_scores = [float(s) for s in route_scores]

            # Best route analysis
            try:
                if hasattr(routes, "reaction_trees"):
                    best_tree = routes.reaction_trees[0]
                    tree_dict = best_tree.to_dict()
                elif hasattr(routes, "trees"):
                    best_tree = routes.trees[0]
                    tree_dict = best_tree if isinstance(best_tree, dict) else {}
                else:
                    tree_dict = {}

                if tree_dict:
                    result.total_steps = count_route_steps(tree_dict)
                    result.longest_linear_sequence = compute_route_lls(tree_dict)
                    result.num_building_blocks = count_route_leaves(tree_dict)
            except Exception:
                pass  # Route detail extraction can fail — score still valid

            result.score = float(route_scores[0]) if route_scores else 0.0
            result.solved = bool(stats.get("is_solved", False))

        # Fallback solved check
        if not result.solved and hasattr(finder, "search_stats"):
            result.solved = bool(
                finder.search_stats.get("is_solved", False)
            )

    except Exception as e:
        result.wall_time_s = time.perf_counter() - t0
        result.error = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AiZynthFinder on USPTO 190"
    )
    parser.add_argument(
        "--datadir", type=Path, default=Path("benchmarks/data"),
        help="Directory with targets.smi and stock.smi"
    )
    parser.add_argument(
        "--targets", type=Path, default=None,
        help="Override: path to targets file"
    )
    parser.add_argument(
        "--stock", type=Path, default=None,
        help="Override: path to stock file"
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory (default: datadir)"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to AiZynthFinder config YAML"
    )
    parser.add_argument(
        "--expansion-model", type=Path, default=None,
        help="Path to expansion model (.onnx / .hdf5)"
    )
    parser.add_argument(
        "--template-file", type=Path, default=None,
        help="Path to template file (.hdf5)"
    )
    parser.add_argument(
        "--time-limit", type=int, default=120,
        help="MCTS time limit per molecule in seconds (default: 120)"
    )
    parser.add_argument(
        "--iteration-limit", type=int, default=100,
        help="MCTS iteration limit per molecule (default: 100)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N targets"
    )
    args = parser.parse_args()

    # --- Check aizynthfinder is installed ---
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
    except ImportError:
        print("Error: aizynthfinder is not installed.")
        print("Install it with: pip install aizynthfinder")
        print("Then download public data: aizynthcli --config config.yml download-public-data")
        sys.exit(1)

    # --- Resolve paths ---
    targets_path = args.targets or (args.datadir / "targets.smi")
    stock_path = args.stock or (args.datadir / "stock.smi")
    outdir = args.outdir or args.datadir

    if not targets_path.exists():
        print(f"Error: targets not found: {targets_path}")
        sys.exit(1)
    if not stock_path.exists():
        print(f"Error: stock not found: {stock_path}")
        sys.exit(1)

    # --- Load targets ---
    targets = [
        l.strip() for l in targets_path.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Loaded {len(targets)} targets from {targets_path}")

    # --- Initialize AiZynthFinder ---
    print("Initializing AiZynthFinder...")
    t0 = time.time()

    if args.config and args.config.exists():
        finder = AiZynthFinder(configfile=str(args.config))
        print(f"  Loaded config from {args.config}")
    elif args.expansion_model and args.template_file:
        config_dict = create_config_dict(
            stock_path=stock_path,
            expansion_model=args.expansion_model,
            template_file=args.template_file,
            time_limit=args.time_limit,
            iteration_limit=args.iteration_limit,
        )
        # Write temp config
        import yaml
        tmp_config = args.datadir / "_aizynthfinder_config.yml"
        with open(tmp_config, "w") as f:
            yaml.dump(config_dict, f)
        finder = AiZynthFinder(configfile=str(tmp_config))
        print(f"  Created config with model={args.expansion_model}")
    else:
        print("Error: must provide either --config or both --expansion-model and --template-file")
        sys.exit(1)

    # Load stock
    try:
        stock_key = list(finder.stock.items.keys())[0] if hasattr(finder.stock, 'items') else None
        if stock_key:
            finder.stock.select(stock_key)
        elif hasattr(finder, 'stock') and hasattr(finder.stock, 'load'):
            finder.stock.load(str(stock_path), "benchmark")
            finder.stock.select("benchmark")
    except Exception as e:
        print(f"  Warning: stock loading issue (may already be loaded via config): {e}")

    init_time = time.time() - t0
    print(f"  Initialized in {init_time:.1f}s")

    # --- Run benchmark ---
    print(f"\nRunning AiZynthFinder benchmark: time_limit={args.time_limit}s, "
          f"iteration_limit={args.iteration_limit}, targets={len(targets)}")

    results: list[AiZynthResult] = []
    solved = 0
    total_time = 0.0
    errors = 0
    n = len(targets)

    for i, smi in enumerate(targets):
        r = run_single_aizynthfinder(finder, smi)
        results.append(r)

        total_time += r.wall_time_s
        if r.error:
            errors += 1
        elif r.solved:
            solved += 1

        if (i + 1) % 10 == 0 or i == n - 1:
            print(
                f"  [{i+1:3d}/{n}] solved={solved} errors={errors} "
                f"avg_time={total_time/(i+1):.2f}s"
            )

    # --- Save results ---
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    times = [r.wall_time_s for r in results]
    solved_steps = [r.total_steps for r in results if r.solved]
    solved_lls = [r.longest_linear_sequence for r in results if r.solved]
    solved_bbs = [r.num_building_blocks for r in results if r.solved]

    def _safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def _safe_median(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return round(s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2, 4)

    output = {
        "tool": "aizynthfinder",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "targets_file": str(targets_path),
            "stock_file": str(stock_path),
            "num_targets": len(targets),
            "time_limit_s": args.time_limit,
            "iteration_limit": args.iteration_limit,
            "config_file": str(args.config) if args.config else None,
        },
        "total_wall_time_s": round(total_time, 2),
        "summary": {
            "solved": solved,
            "solved_rate": round(solved / n, 4) if n else 0,
            "errors": errors,
            "mean_time_s": _safe_mean(times),
            "median_time_s": _safe_median(times),
            "total_time_s": round(sum(times), 2),
            "mean_steps": _safe_mean(solved_steps),
            "median_steps": _safe_median(solved_steps),
            "mean_lls": _safe_mean(solved_lls),
            "mean_bbs": _safe_mean(solved_bbs),
        },
        "molecules": [r.to_dict() for r in results],
    }

    out_path = outdir / "results_aizynthfinder.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AiZynthFinder benchmark complete")
    print(f"  Solved: {solved}/{n} ({100*solved/n:.1f}%)")
    print(f"  Mean time: {_safe_mean(times):.2f}s/mol")
    print(f"  Results saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
