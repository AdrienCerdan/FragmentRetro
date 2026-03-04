#!/usr/bin/env python
"""FragmentRetro T3 Retrosynthesis CLI.

Production-grade tool for synthetic accessibility analysis and retrosynthesis
using the SMARTS-based T3 engine. Supports single SMILES, batch .smi files,
and .csv inputs with optional strict synthesis filters and route visualization.

Usage:
    # Single molecule
    python retro.py --smiles "c1ccc(-c2ccccc2)cc1" --stock mol_properties.json

    # Batch file
    python retro.py --input targets.smi --stock mol_properties.json

    # CSV with column selection
    python retro.py --input targets.csv --smiles-column SMILES --stock mol_properties.json

    # With strict filters and route figures
    python retro.py --smiles "CCO" --stock mol_properties.json --strict --figures

    # With constraints
    python retro.py --input targets.smi --stock mol_properties.json \
        --max-steps 3 --blocked-reactions Grignard --min-reliability 0.8

    # Parallel processing
    python retro.py --input targets.smi --stock mol_properties.json -j 8

    # CSV output
    python retro.py --input targets.smi --stock mol_properties.json --format csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem

from fragmentretro.constraints import ConstraintConfig, build_constraints
from fragmentretro.reaction_library import ReactionLibrary
from fragmentretro.scoring import load_compound_filter
from fragmentretro.smarts_retro import RetroSynthNode, SmartsRetrosynthesis
from fragmentretro.utils.logging_config import logger, setup_logging

setup_logging()


# ---------------------------------------------------------------------------
# Multiprocessing globals
# ---------------------------------------------------------------------------
_worker_cf = None
_worker_lib = None
_worker_opts = None


def _init_worker(mol_props_path, fp_size, opts):
    """Initialize per-worker globals."""
    global _worker_cf, _worker_lib, _worker_opts
    _worker_cf = load_compound_filter(mol_props_path, fpSize=fp_size)
    _worker_lib = ReactionLibrary.default()
    _worker_opts = opts


# ---------------------------------------------------------------------------
# Core retrosynthesis function
# ---------------------------------------------------------------------------

def run_retro_single(
    smiles: str,
    cf,
    lib: ReactionLibrary,
    max_routes: int = 5,
    max_depth: int = 5,
    max_nodes: int = 2000,
    strict: bool = False,
    constraints: ConstraintConfig | None = None,
) -> dict:
    """Run T3 retrosynthesis on a single SMILES.

    Returns a result dict with solved, score, routes, and timing.
    """
    t0 = time.time()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _error_result(smiles, t0, "Invalid SMILES")
        canonical = Chem.MolToSmiles(mol)

        # Build synthesis filters
        synthesis_filters = None
        if strict:
            from fragmentretro.synthesis_filters import SynthesisFilters
            synthesis_filters = SynthesisFilters()

        retro = SmartsRetrosynthesis(
            library=lib,
            max_depth=max_depth,
            max_nodes=max_nodes,
            constraints=constraints,
            min_bb_heavy_atoms=getattr(cf, "min_heavy_atoms", 0),
            max_bb_heavy_atoms=getattr(cf, "max_heavy_atoms", 100),
            synthesis_filters=synthesis_filters,
        )

        def is_purchasable(smi):
            try:
                return cf.has_match(smi)
            except Exception:
                return False

        routes = retro.retrosynthesise(
            canonical, is_purchasable=is_purchasable, max_routes=max_routes
        )

        if not routes:
            return {
                "smiles": canonical,
                "solved": False,
                "score": 0.0,
                "n_routes": 0,
                "best_route": None,
                "routes": [],
                "time_s": round(time.time() - t0, 4),
            }

        # Sort routes: solved first, then by reliability
        routes.sort(key=lambda r: (r.is_solved, r.avg_reliability), reverse=True)
        routes = routes[:max_routes]

        best = routes[0]

        # Compute score
        if best.is_solved:
            reliability_score = best.avg_reliability
            steps = best.num_steps
            efficiency_score = 1.0 / (1.0 + 0.2 * steps)
            score = 0.5 * reliability_score + 0.5 * efficiency_score
        else:
            total_leaves = best.num_leaves
            bb_leaves = sum(1 for n in _iter_leaves(best) if n.is_building_block)
            partial = bb_leaves / total_leaves if total_leaves > 0 else 0.0
            score = 0.3 * partial * best.avg_reliability

        routes_info = []
        for r in routes:
            routes_info.append({
                "solved": r.is_solved,
                "steps": r.num_steps,
                "lls": r.longest_linear_sequence,
                "n_bbs": r.num_leaves,
                "avg_reliability": round(r.avg_reliability, 4),
                "reaction": r.reaction.name if r.reaction else None,
                "tree": r.to_dict(),
            })

        return {
            "smiles": canonical,
            "solved": best.is_solved,
            "score": round(score, 6),
            "n_routes": len(routes_info),
            "best_route": {
                "solved": best.is_solved,
                "steps": best.num_steps,
                "lls": best.longest_linear_sequence,
                "n_bbs": best.num_leaves,
                "avg_reliability": round(best.avg_reliability, 4),
            },
            "routes": routes_info,
            "_route_objects": routes,  # kept for figure generation, stripped on output
            "time_s": round(time.time() - t0, 4),
        }

    except Exception as e:
        return _error_result(smiles, t0, str(e))


def _error_result(smiles, t0, error_msg):
    return {
        "smiles": smiles,
        "solved": False,
        "score": 0.0,
        "n_routes": 0,
        "best_route": None,
        "routes": [],
        "time_s": round(time.time() - t0, 4),
        "error": error_msg,
    }


def _iter_leaves(node):
    if node.is_leaf:
        yield node
    else:
        for c in node.children:
            yield from _iter_leaves(c)


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _process_molecule(task):
    """Process a single molecule for multiprocessing."""
    idx, smiles, n_total = task
    opts = _worker_opts

    result = run_retro_single(
        smiles,
        _worker_cf,
        _worker_lib,
        max_routes=opts["max_routes"],
        max_depth=opts["max_depth"],
        max_nodes=opts["max_nodes"],
        strict=opts["strict"],
        constraints=opts.get("constraints"),
    )
    result["idx"] = idx
    return result


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def load_smiles(args) -> list[str]:
    """Load SMILES from CLI arguments."""
    if args.smiles:
        return [args.smiles]

    if not args.input:
        print("Error: provide --smiles or --input", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() == ".csv":
        return _load_csv(input_path, args.smiles_column)
    else:
        return _load_smi(input_path)


def _load_smi(path: Path) -> list[str]:
    """Load from .smi or plain text (one SMILES per line)."""
    smiles_list = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                smiles_list.append(parts[0])
    return smiles_list


def _load_csv(path: Path, column: str = "smiles") -> list[str]:
    """Load SMILES from a CSV file."""
    smiles_list = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            # Try case-insensitive match
            col_map = {c.lower(): c for c in reader.fieldnames}
            if column.lower() in col_map:
                column = col_map[column.lower()]
            else:
                avail = ", ".join(reader.fieldnames)
                print(f"Error: column '{column}' not found. Available: {avail}",
                      file=sys.stderr)
                sys.exit(1)
        for row in reader:
            smi = row[column].strip()
            if smi:
                smiles_list.append(smi)
    return smiles_list


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _strip_internal_keys(result: dict) -> dict:
    """Remove internal keys before serialization."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def write_json_output(results: list[dict], summary: dict, path: Path):
    """Write full JSON output."""
    clean_results = [_strip_internal_keys(r) for r in results]
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": clean_results}, f, indent=2)


def write_csv_output(results: list[dict], path: Path):
    """Write CSV summary output."""
    fields = [
        "smiles", "solved", "score", "n_routes",
        "steps", "lls", "n_bbs", "avg_reliability", "time_s", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            br = r.get("best_route") or {}
            writer.writerow({
                "smiles": r["smiles"],
                "solved": r["solved"],
                "score": r["score"],
                "n_routes": r["n_routes"],
                "steps": br.get("steps", ""),
                "lls": br.get("lls", ""),
                "n_bbs": br.get("n_bbs", ""),
                "avg_reliability": br.get("avg_reliability", ""),
                "time_s": r["time_s"],
                "error": r.get("error", ""),
            })


def print_text_result(result: dict, verbose: bool = False):
    """Print a single result to stdout."""
    smi = result["smiles"]
    solved = "✓" if result["solved"] else "✗"
    score = result["score"]
    n_routes = result["n_routes"]
    time_s = result["time_s"]
    error = result.get("error")

    if error:
        print(f"  {solved} {smi[:60]:60s}  ERROR: {error}")
        return

    br = result.get("best_route") or {}
    steps = br.get("steps", "-")
    lls = br.get("lls", "-")
    n_bbs = br.get("n_bbs", "-")
    rel = br.get("avg_reliability", 0)

    print(
        f"  {solved} {smi[:60]:60s}  "
        f"score={score:.3f}  routes={n_routes}  "
        f"steps={steps}  lls={lls}  bbs={n_bbs}  "
        f"rel={rel:.3f}  ({time_s:.2f}s)"
    )

    if verbose and result.get("_route_objects"):
        best = result["_route_objects"][0]
        print(best.pretty_print())
        print()


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics."""
    valid = [r for r in results if "error" not in r]
    n_total = len(results)
    n_valid = len(valid)
    n_solved = sum(1 for r in valid if r["solved"])
    scores = [r["score"] for r in valid]
    times = [r["time_s"] for r in valid]

    summary = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_solved": n_solved,
        "solve_rate": round(n_solved / n_valid, 4) if n_valid else 0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "median_score": round(sorted(scores)[len(scores) // 2], 4) if scores else 0,
        "mean_time_s": round(sum(times) / len(times), 4) if times else 0,
        "total_time_s": round(sum(times), 2),
    }

    solved_results = [r for r in valid if r["solved"] and r.get("best_route")]
    if solved_results:
        steps_list = [r["best_route"]["steps"] for r in solved_results]
        lls_list = [r["best_route"]["lls"] for r in solved_results]
        bbs_list = [r["best_route"]["n_bbs"] for r in solved_results]
        rels = [r["best_route"]["avg_reliability"] for r in solved_results]
        summary["solved_metrics"] = {
            "mean_steps": round(sum(steps_list) / len(steps_list), 2),
            "mean_lls": round(sum(lls_list) / len(lls_list), 2),
            "mean_bbs": round(sum(bbs_list) / len(bbs_list), 2),
            "mean_reliability": round(sum(rels) / len(rels), 4),
        }

    n_errors = sum(1 for r in results if "error" in r)
    if n_errors:
        summary["n_errors"] = n_errors

    return summary


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(results: list[dict], output_dir: Path, max_routes_to_draw: int = 3):
    """Generate route figures for all results."""
    from fragmentretro.t3_visualizer import draw_t3_route

    output_dir.mkdir(parents=True, exist_ok=True)
    n_drawn = 0

    for result in results:
        routes = result.get("_route_objects", [])
        if not routes:
            continue

        smi = result["smiles"]
        # Create a safe filename from the SMILES
        safe_name = smi[:40].replace("/", "_").replace("\\", "_").replace(
            "(", "").replace(")", "").replace("[", "").replace("]", "").replace(
            "#", "").replace("=", "").replace(".", "_")

        for i, route in enumerate(routes[:max_routes_to_draw]):
            fname = f"{safe_name}_route{i:02d}.png"
            path = output_dir / fname
            try:
                draw_t3_route(route, output_path=path)
                n_drawn += 1
            except Exception as e:
                logger.warning(f"Failed to draw route for {smi}: {e}")

    return n_drawn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FragmentRetro T3 Retrosynthesis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single molecule
  python retro.py --smiles "c1ccc(-c2ccccc2)cc1" --stock mol_properties.json

  # Batch with strict filters and figures
  python retro.py --input targets.smi --stock mol_properties.json --strict --figures

  # CSV output with constraints
  python retro.py --input targets.csv --stock mol_properties.json \\
      --format csv --max-steps 3 --blocked-reactions Grignard
        """,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--smiles", "-s", type=str,
                              help="Single SMILES string to analyze")
    input_group.add_argument("--input", "-i", type=str,
                              help="Input file (.smi or .csv)")
    parser.add_argument("--smiles-column", type=str, default="smiles",
                        help="Column name for SMILES in CSV (default: smiles)")

    # Stock
    parser.add_argument("--stock", type=str, default=None,
                        help="Path to mol_properties.json (building block catalog)")

    # Search parameters
    parser.add_argument("--max-routes", type=int, default=5,
                        help="Max routes per molecule (default: 5)")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Max retrosynthesis tree depth (default: 5)")
    parser.add_argument("--max-nodes", type=int, default=2000,
                        help="Max nodes to explore per molecule (default: 2000)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Timeout per molecule in seconds (default: 120)")

    # Constraints
    constraint_group = parser.add_argument_group("Constraints")
    constraint_group.add_argument("--strict", action="store_true",
                                   help="Enable strict synthesis filters (Lilly, FG, Sterics)")
    constraint_group.add_argument("--max-steps", type=int, default=None,
                                   help="Max synthetic steps in a route")
    constraint_group.add_argument("--allowed-reactions", nargs="+", default=None,
                                   help="Whitelist reaction names (partial, case-insensitive)")
    constraint_group.add_argument("--blocked-reactions", nargs="+", default=None,
                                   help="Blacklist reaction names")
    constraint_group.add_argument("--allowed-classes", nargs="+", default=None,
                                   help="Whitelist reaction classes")
    constraint_group.add_argument("--min-reliability", type=float, default=0.0,
                                   help="Min per-reaction reliability (default: 0.0)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", type=str, default=None,
                               help="Output file path (default: results.json or results.csv)")
    output_group.add_argument("--format", "-f", type=str, default="json",
                               choices=["json", "csv", "text"],
                               help="Output format (default: json)")
    output_group.add_argument("--figures", action="store_true",
                               help="Generate route figures (PNG)")
    output_group.add_argument("--figures-dir", type=str, default="./routes",
                               help="Directory for route figures (default: ./routes)")
    output_group.add_argument("--max-figures", type=int, default=3,
                               help="Max routes to draw per molecule (default: 3)")
    output_group.add_argument("--verbose", "-v", action="store_true",
                               help="Verbose output with route tree pretty-printing")

    # Parallelism
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Parallel workers (default: 1, 0 = auto)")

    # Utility
    parser.add_argument("--list-reactions", action="store_true",
                        help="List all available reactions and exit")
    parser.add_argument("--fp-size", type=int, default=2048,
                        help="Fingerprint size (default: 2048)")

    args = parser.parse_args()

    # --- List reactions mode ---
    if args.list_reactions:
        lib = ReactionLibrary.default()
        print(f"\n{'ID':6s}  {'Name':45s}  {'Class':20s}  {'Reliability':12s}  Source")
        print("-" * 100)
        for rxn in sorted(lib.reactions, key=lambda r: (r.reaction_class, r.name)):
            print(f"{rxn.id:6s}  {rxn.name:45s}  {rxn.reaction_class:20s}  "
                  f"{rxn.reliability:12.3f}  {rxn.source}")
        print(f"\nTotal: {len(lib)} reactions")
        return

    # --- Load targets ---
    if not args.stock:
        print("Error: --stock is required when running analysis", file=sys.stderr)
        sys.exit(1)

    targets = load_smiles(args)
    if not targets:
        print("Error: no SMILES loaded", file=sys.stderr)
        sys.exit(1)

    # --- Build constraints ---
    constraints = build_constraints(
        allowed_reactions=args.allowed_reactions,
        blocked_reactions=args.blocked_reactions,
        allowed_classes=args.allowed_classes,
        max_steps=args.max_steps,
        min_reliability=args.min_reliability,
    )

    # --- Determine output path ---
    if args.output:
        output_path = Path(args.output)
    elif args.format == "csv":
        output_path = Path("results.csv")
    elif args.format == "json":
        output_path = Path("results.json")
    else:
        output_path = None  # text mode, stdout only

    # --- Worker options ---
    opts = {
        "max_routes": args.max_routes,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "strict": args.strict,
        "constraints": constraints,
    }

    n_workers = args.workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    if n_workers < 0:
        n_workers = 1

    # --- Header ---
    print(f"\n{'='*70}")
    print(f"  FragmentRetro T3 Retrosynthesis")
    print(f"{'='*70}")
    print(f"  Targets:     {len(targets)}")
    print(f"  Stock:       {args.stock}")
    print(f"  Max depth:   {args.max_depth}")
    print(f"  Max nodes:   {args.max_nodes}")
    print(f"  Max routes:  {args.max_routes}")
    print(f"  Strict:      {'Yes' if args.strict else 'No'}")
    if constraints:
        if constraints.allowed_reactions:
            print(f"  Allowed rxn: {', '.join(constraints.allowed_reactions)}")
        if constraints.blocked_reactions:
            print(f"  Blocked rxn: {', '.join(constraints.blocked_reactions)}")
        if constraints.allowed_classes:
            print(f"  Allowed cls: {', '.join(constraints.allowed_classes)}")
        if constraints.max_steps:
            print(f"  Max steps:   {constraints.max_steps}")
        if constraints.min_reliability > 0:
            print(f"  Min reliab:  {constraints.min_reliability}")
    print(f"  Workers:     {n_workers}" + (" (parallel)" if n_workers > 1 else " (serial)"))
    print(f"  Output:      {output_path or 'stdout'} ({args.format})")
    if args.figures:
        print(f"  Figures:     {args.figures_dir}")
    print(f"{'='*70}\n")

    # --- Run ---
    t_start = time.time()

    if n_workers > 1:
        results = _run_parallel(targets, args, opts, n_workers)
    else:
        results = _run_serial(targets, args, opts)

    total_time = time.time() - t_start

    # --- Summary ---
    summary = compute_summary(results)
    summary["total_wall_time_s"] = round(total_time, 2)
    summary["settings"] = {
        "strict": args.strict,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "max_routes": args.max_routes,
        "workers": n_workers,
    }
    if constraints:
        summary["settings"]["constraints"] = {
            "allowed_reactions": constraints.allowed_reactions,
            "blocked_reactions": constraints.blocked_reactions,
            "allowed_classes": constraints.allowed_classes,
            "max_steps": constraints.max_steps,
            "min_reliability": constraints.min_reliability,
        }

    # --- Figures ---
    if args.figures:
        fig_dir = Path(args.figures_dir)
        print(f"\nGenerating route figures in {fig_dir} ...")
        n_drawn = generate_figures(results, fig_dir, max_routes_to_draw=args.max_figures)
        print(f"  {n_drawn} figures generated")

    # --- Output ---
    if args.format == "json" and output_path:
        write_json_output(results, summary, output_path)
        print(f"\nJSON results saved to: {output_path}")
    elif args.format == "csv" and output_path:
        write_csv_output(results, output_path)
        print(f"\nCSV results saved to: {output_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Molecules:    {summary['n_total']}")
    print(f"  Valid:         {summary['n_valid']}")
    print(f"  Solved:        {summary['n_solved']} ({summary['solve_rate']:.1%})")
    print(f"  Mean score:    {summary['mean_score']:.4f}")
    print(f"  Mean time:     {summary['mean_time_s']:.3f} s/mol")
    print(f"  Total time:    {summary['total_wall_time_s']:.1f} s")
    if "solved_metrics" in summary:
        sm = summary["solved_metrics"]
        print(f"  Solved routes:")
        print(f"    Mean steps:       {sm['mean_steps']:.1f}")
        print(f"    Mean LLS:         {sm['mean_lls']:.1f}")
        print(f"    Mean BBs:         {sm['mean_bbs']:.1f}")
        print(f"    Mean reliability: {sm['mean_reliability']:.4f}")
    if "n_errors" in summary:
        print(f"  Errors:        {summary['n_errors']}")
    print(f"{'='*70}\n")


def _run_serial(targets, args, opts):
    """Serial execution."""
    print(f"Loading CompoundFilter from {args.stock} ...")
    cf = load_compound_filter(args.stock, fpSize=args.fp_size)
    print(f"  {cf.len_BBs} building blocks loaded")

    print("Loading ReactionLibrary ...")
    lib = ReactionLibrary.default()
    print(f"  {len(lib)} reactions loaded\n")

    constraints = opts.get("constraints")
    results = []
    n_total = len(targets)

    for idx, smiles in enumerate(targets):
        result = run_retro_single(
            smiles, cf, lib,
            max_routes=opts["max_routes"],
            max_depth=opts["max_depth"],
            max_nodes=opts["max_nodes"],
            strict=opts["strict"],
            constraints=constraints,
        )
        result["idx"] = idx
        results.append(result)
        print_text_result(result, verbose=args.verbose)

    return results


def _run_parallel(targets, args, opts, n_workers):
    """Parallel execution with multiprocessing."""
    import multiprocessing as mp

    print(f"Parallel mode: {n_workers} workers")
    print(f"Loading CompoundFilter + ReactionLibrary in each worker ...\n")

    n_total = len(targets)
    tasks = [(idx, smi, n_total) for idx, smi in enumerate(targets)]

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(args.stock, args.fp_size, opts),
    ) as pool:
        results = []
        for result in pool.imap_unordered(_process_molecule, tasks, chunksize=4):
            results.append(result)
            print_text_result(result, verbose=args.verbose)

    # Re-sort by original index
    results.sort(key=lambda r: r.get("idx", 0))
    return results


if __name__ == "__main__":
    main()
