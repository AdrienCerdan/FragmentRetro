#!/usr/bin/env python
"""Benchmark FragmentRetro on USPTO 190 targets.

Runs all scoring tiers and collects per-molecule timing, accuracy,
and route quality metrics. Results are saved as JSON for comparison
with AiZynthFinder.

Usage:
    python run_fragmentretro.py --datadir benchmarks/data

    # Run only specific tiers
    python run_fragmentretro.py --datadir benchmarks/data --tiers 1 2

    # Limit to first N targets (for quick testing)
    python run_fragmentretro.py --datadir benchmarks/data --limit 10

    # Adjust timeout and SMARTS search depth
    python run_fragmentretro.py --datadir benchmarks/data \
        --timeout 120 --smarts-depth 3 --smarts-nodes 1000
"""

import argparse
import json
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fragmentretro.scoring import (
    compute_score,
    compute_score_with_dag,
    compute_score_validated,
    compute_score_smarts,
    load_compound_filter,
    DAGScoreResult,
)
from fragmentretro.reaction_library import ReactionLibrary
from fragmentretro.constraints import ConstraintConfig
from fragmentretro.utils.logging_config import logger


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MoleculeResult:
    """Benchmark result for a single molecule."""
    smiles: str
    tier: int

    # Timing
    wall_time_s: float = 0.0
    timed_out: bool = False
    error: str | None = None

    # Feasibility
    solved: bool = False
    score: float = 0.0

    # Route quality (Tier 1b / 2 / 3)
    total_steps: int = 0
    longest_linear_sequence: int = 0
    num_building_blocks: int = 0
    convergence_score: float = 0.0

    # SMARTS-specific (Tier 2)
    matched_reactions: list[str] = field(default_factory=list)
    validation_coverage: float = 0.0
    constraints_satisfied: bool = True

    # Component scores (Tier 1b / 2)
    step_score: float = 0.0
    availability_score: float = 0.0
    bond_feasibility_score: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "smiles": self.smiles,
            "tier": self.tier,
            "wall_time_s": round(self.wall_time_s, 4),
            "timed_out": self.timed_out,
            "error": self.error,
            "solved": self.solved,
            "score": round(self.score, 6),
            "total_steps": self.total_steps,
            "longest_linear_sequence": self.longest_linear_sequence,
            "num_building_blocks": self.num_building_blocks,
            "convergence_score": round(self.convergence_score, 4),
        }
        if self.tier == 2:
            d.update({
                "matched_reactions": self.matched_reactions,
                "validation_coverage": round(self.validation_coverage, 4),
                "constraints_satisfied": self.constraints_satisfied,
            })
        if self.tier in (1, 2):
            d.update({
                "step_score": round(self.step_score, 4),
                "availability_score": round(self.availability_score, 4),
                "bond_feasibility_score": round(self.bond_feasibility_score, 4),
            })
        return d


# ---------------------------------------------------------------------------
# Scoring runners per tier
# ---------------------------------------------------------------------------

def run_tier1(smiles: str, cf, timeout: float) -> MoleculeResult:
    """Tier 1: BRICS-only (continuous score + DAG)."""
    result = MoleculeResult(smiles=smiles, tier=1)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))

    try:
        t0 = time.perf_counter()
        dag_result = compute_score_with_dag(smiles, cf)
        result.wall_time_s = time.perf_counter() - t0

        result.score = dag_result.score
        result.solved = dag_result.feasible and dag_result.score > 0
        result.total_steps = dag_result.total_steps
        result.longest_linear_sequence = dag_result.longest_linear_sequence
        result.num_building_blocks = dag_result.num_building_blocks
        result.convergence_score = dag_result.convergence_score
        result.step_score = dag_result.step_score
        result.availability_score = dag_result.availability_score
        result.bond_feasibility_score = dag_result.bond_feasibility_score

    except TimeoutError:
        result.wall_time_s = timeout
        result.timed_out = True
    except Exception as e:
        result.wall_time_s = time.perf_counter() - t0
        result.error = str(e)
    finally:
        signal.alarm(0)

    return result


def run_tier2(smiles: str, cf, lib: ReactionLibrary, timeout: float) -> MoleculeResult:
    """Tier 2: BRICS + SMARTS validation."""
    result = MoleculeResult(smiles=smiles, tier=2)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))

    try:
        t0 = time.perf_counter()
        dag_result = compute_score_validated(smiles, cf, lib)
        result.wall_time_s = time.perf_counter() - t0

        result.score = dag_result.score
        result.solved = dag_result.feasible and dag_result.score > 0
        result.total_steps = dag_result.total_steps
        result.longest_linear_sequence = dag_result.longest_linear_sequence
        result.num_building_blocks = dag_result.num_building_blocks
        result.convergence_score = dag_result.convergence_score
        result.step_score = dag_result.step_score
        result.availability_score = dag_result.availability_score
        result.bond_feasibility_score = dag_result.bond_feasibility_score
        result.matched_reactions = [name for name, _ in dag_result.matched_reactions]
        result.validation_coverage = dag_result.validation_coverage
        result.constraints_satisfied = dag_result.constraints_satisfied

    except TimeoutError:
        result.wall_time_s = timeout
        result.timed_out = True
    except Exception as e:
        result.wall_time_s = time.perf_counter() - t0
        result.error = str(e)
    finally:
        signal.alarm(0)

    return result


def run_tier3(
    smiles: str, cf, lib: ReactionLibrary,
    timeout: float, max_depth: int, max_nodes: int,
) -> MoleculeResult:
    """Tier 3: Standalone SMARTS retrosynthesis."""
    result = MoleculeResult(smiles=smiles, tier=3)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout))

    try:
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        t0 = time.perf_counter()

        # Run SMARTS retrosynthesis directly (not via compute_score_smarts)
        # so we can capture route details
        retro = SmartsRetrosynthesis(lib, max_depth=max_depth, max_nodes=max_nodes)
        routes = retro.retrosynthesise(
            smiles,
            is_purchasable=lambda smi: cf.has_match(smi),
            max_routes=5,
        )
        result.wall_time_s = time.perf_counter() - t0

        if routes:
            best = max(routes, key=lambda r: (r.is_solved, r.avg_reliability))
            result.solved = best.is_solved
            result.total_steps = best.num_steps
            result.longest_linear_sequence = best.longest_linear_sequence
            result.num_building_blocks = best.num_leaves
            result.convergence_score = best.avg_reliability

            # Score: same formula as compute_score_smarts
            if best.is_solved:
                rel = best.avg_reliability
                eff = 1.0 / (1.0 + 0.2 * best.num_steps)
                result.score = 0.5 * rel + 0.5 * eff
            else:
                total = best.num_leaves
                bb_count = sum(
                    1 for n in _iter_leaves(best) if n.is_building_block
                )
                partial = bb_count / total if total > 0 else 0.0
                result.score = 0.3 * partial * best.avg_reliability

            result.matched_reactions = []
            _collect_reactions(best, result.matched_reactions)

    except TimeoutError:
        result.wall_time_s = timeout
        result.timed_out = True
    except Exception as e:
        result.wall_time_s = time.perf_counter() - t0
        result.error = str(e)
    finally:
        signal.alarm(0)

    return result


def _iter_leaves(node):
    """Yield leaf nodes from a RetroSynthNode."""
    if node.is_leaf:
        yield node
    else:
        for c in node.children:
            yield from _iter_leaves(c)


def _collect_reactions(node, acc: list):
    """Collect reaction names from a RetroSynthNode tree."""
    if node.reaction:
        acc.append(node.reaction.name)
    for c in node.children:
        _collect_reactions(c, acc)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    targets: list[str],
    cf,
    lib: ReactionLibrary,
    tiers: list[int],
    timeout: float,
    smarts_depth: int,
    smarts_nodes: int,
) -> dict[int, list[MoleculeResult]]:
    """Run benchmark across all tiers and targets."""

    results: dict[int, list[MoleculeResult]] = {t: [] for t in tiers}
    n = len(targets)

    for tier in tiers:
        print(f"\n{'='*60}")
        print(f"  Tier {tier}: {'BRICS + DAG' if tier == 1 else 'BRICS + SMARTS' if tier == 2 else 'SMARTS-only'}")
        print(f"{'='*60}")

        solved = 0
        total_time = 0.0
        timeouts = 0
        errors = 0

        for i, smi in enumerate(targets):
            if tier == 1:
                r = run_tier1(smi, cf, timeout)
            elif tier == 2:
                r = run_tier2(smi, cf, lib, timeout)
            elif tier == 3:
                r = run_tier3(smi, cf, lib, timeout, smarts_depth, smarts_nodes)
            else:
                continue

            results[tier].append(r)
            total_time += r.wall_time_s
            if r.timed_out:
                timeouts += 1
            elif r.error:
                errors += 1
            elif r.solved:
                solved += 1

            # Progress
            if (i + 1) % 10 == 0 or i == n - 1:
                print(
                    f"  [{i+1:3d}/{n}] solved={solved} timeouts={timeouts} "
                    f"errors={errors} avg_time={total_time/(i+1):.2f}s"
                )

        print(f"\n  Tier {tier} summary:")
        print(f"    Solved:   {solved}/{n} ({100*solved/n:.1f}%)")
        print(f"    Timeouts: {timeouts}/{n}")
        print(f"    Errors:   {errors}/{n}")
        print(f"    Total:    {total_time:.1f}s  (avg {total_time/n:.2f}s/mol)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FragmentRetro (all tiers) on USPTO 190"
    )
    parser.add_argument(
        "--datadir", type=Path, default=Path("benchmarks/data"),
        help="Directory with targets.smi and mol_properties.json"
    )
    parser.add_argument(
        "--targets", type=Path, default=None,
        help="Override: path to targets file (default: datadir/targets.smi)"
    )
    parser.add_argument(
        "--mol-properties", type=Path, default=None,
        help="Override: path to mol_properties.json"
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory for results (default: datadir)"
    )
    parser.add_argument(
        "--tiers", type=int, nargs="+", default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which tiers to run (default: 1 2 3)"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Timeout per molecule in seconds (default: 60)"
    )
    parser.add_argument(
        "--smarts-depth", type=int, default=3,
        help="Tier 3: max retrosynthesis tree depth (default: 3)"
    )
    parser.add_argument(
        "--smarts-nodes", type=int, default=500,
        help="Tier 3: max nodes explored (default: 500)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to first N targets (for testing)"
    )
    parser.add_argument(
        "--fp-size", type=int, default=2048,
        help="Fingerprint size (default: 2048)"
    )
    args = parser.parse_args()

    # Resolve paths
    targets_path = args.targets or (args.datadir / "targets.smi")
    mol_props_path = args.mol_properties or (args.datadir / "mol_properties.json")
    outdir = args.outdir or args.datadir

    for p, label in [(targets_path, "targets"), (mol_props_path, "mol_properties")]:
        if not p.exists():
            print(f"Error: {label} file not found: {p}")
            print("Run prepare_data.py first.")
            sys.exit(1)

    # Load targets
    targets = [
        l.strip() for l in targets_path.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Loaded {len(targets)} targets from {targets_path}")

    # Load CompoundFilter
    print(f"Loading CompoundFilter from {mol_props_path}...")
    t0 = time.time()
    cf = load_compound_filter(str(mol_props_path), args.fp_size)
    print(f"  Loaded {cf.len_BBs} BBs in {time.time()-t0:.1f}s")

    # Load ReactionLibrary (for Tier 2 / 3)
    lib = None
    if 2 in args.tiers or 3 in args.tiers:
        print("Loading ReactionLibrary (Hartenfeller + eXplore)...")
        lib = ReactionLibrary.default()
        print(f"  Loaded {len(lib)} reactions")

    # Run benchmark
    print(f"\nRunning FragmentRetro benchmark: tiers={args.tiers}, "
          f"timeout={args.timeout}s, targets={len(targets)}")

    total_t0 = time.time()
    results = run_benchmark(
        targets, cf, lib, args.tiers, args.timeout,
        args.smarts_depth, args.smarts_nodes,
    )
    total_elapsed = time.time() - total_t0

    # Save results
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output = {
        "tool": "fragmentretro",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "targets_file": str(targets_path),
            "mol_properties_file": str(mol_props_path),
            "num_targets": len(targets),
            "num_stock_bbs": cf.len_BBs,
            "timeout_s": args.timeout,
            "tiers": args.tiers,
            "smarts_depth": args.smarts_depth,
            "smarts_nodes": args.smarts_nodes,
            "fp_size": args.fp_size,
        },
        "total_wall_time_s": round(total_elapsed, 2),
        "results": {},
    }

    for tier, tier_results in results.items():
        tier_dicts = [r.to_dict() for r in tier_results]

        # Aggregate stats
        solved = sum(1 for r in tier_results if r.solved)
        timeouts = sum(1 for r in tier_results if r.timed_out)
        errors = sum(1 for r in tier_results if r.error)
        times = [r.wall_time_s for r in tier_results]
        solved_steps = [r.total_steps for r in tier_results if r.solved]
        solved_lls = [r.longest_linear_sequence for r in tier_results if r.solved]
        solved_bbs = [r.num_building_blocks for r in tier_results if r.solved]

        def _safe_mean(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        def _safe_median(lst):
            if not lst:
                return 0.0
            s = sorted(lst)
            n = len(s)
            return round(s[n // 2] if n % 2 else (s[n//2 - 1] + s[n//2]) / 2, 4)

        output["results"][f"tier_{tier}"] = {
            "summary": {
                "solved": solved,
                "solved_rate": round(solved / len(targets), 4) if targets else 0,
                "timeouts": timeouts,
                "errors": errors,
                "mean_time_s": _safe_mean(times),
                "median_time_s": _safe_median(times),
                "total_time_s": round(sum(times), 2),
                "mean_steps": _safe_mean(solved_steps),
                "median_steps": _safe_median(solved_steps),
                "mean_lls": _safe_mean(solved_lls),
                "mean_bbs": _safe_mean(solved_bbs),
            },
            "molecules": tier_dicts,
        }

    out_path = outdir / "results_fragmentretro.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results saved to {out_path}")
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
