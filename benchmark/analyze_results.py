#!/usr/bin/env python
"""Analyze and compare benchmark results across tools.

Reads results from FragmentRetro and AiZynthFinder runs, computes
comparative statistics, and generates tables and plots.

Outputs:
  - Console summary table
  - CSV comparison table
  - Matplotlib plots (solve rate, speed, route quality)
  - LaTeX-formatted table for papers

Usage:
    python analyze_results.py \\
        --config ./benchmark_data/benchmark_config.json \\
        --fr-results ./benchmark_data/results_fragmentretro.json \\
        --aizf-results ./benchmark_data/results_aizynthfinder.json \\
        --output-dir ./benchmark_data/analysis
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("(matplotlib not installed — skipping plots)")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_comparison_table(fr_data: dict, aizf_data: dict) -> list[dict]:
    """Build per-molecule comparison across all tools/tiers."""
    # Index AiZynthFinder results by SMILES
    aizf_by_smi = {}
    for r in aizf_data.get("results", []):
        if r.get("valid"):
            aizf_by_smi[r["smiles"]] = r.get("aizynthfinder", {})

    rows = []
    for r in fr_data.get("results", []):
        if not r.get("valid"):
            continue

        smi = r["smiles"]
        row = {
            "smiles": smi,
            "num_heavy_atoms": r.get("num_heavy_atoms", 0),
            "num_rings": r.get("num_rings", 0),
        }

        # FragmentRetro tiers
        tiers = r.get("tiers", {})

        # Tier 1 binary
        t1b = tiers.get("1_binary", {})
        row["fr_t1b_solved"] = t1b.get("solved", False)
        row["fr_t1b_time"] = t1b.get("time_s", 0)

        # Tier 1 continuous
        t1c = tiers.get("1_continuous", {})
        row["fr_t1c_score"] = t1c.get("score", 0)
        row["fr_t1c_time"] = t1c.get("time_s", 0)

        # Tier 1 DAG
        t1d = tiers.get("1_dag", {})
        row["fr_t1d_solved"] = t1d.get("solved", False)
        row["fr_t1d_score"] = t1d.get("score", 0)
        row["fr_t1d_steps"] = t1d.get("total_steps", 0)
        row["fr_t1d_lls"] = t1d.get("lls", 0)
        row["fr_t1d_bbs"] = t1d.get("num_bbs", 0)
        row["fr_t1d_time"] = t1d.get("time_s", 0)

        # Tier 2
        t2 = tiers.get("2_validated", {})
        row["fr_t2_solved"] = t2.get("solved", False)
        row["fr_t2_score"] = t2.get("score", 0)
        row["fr_t2_steps"] = t2.get("total_steps", 0)
        row["fr_t2_lls"] = t2.get("lls", 0)
        row["fr_t2_bbs"] = t2.get("num_bbs", 0)
        row["fr_t2_validation"] = t2.get("validation_coverage", 0)
        row["fr_t2_time"] = t2.get("time_s", 0)

        # Tier 3
        t3 = tiers.get("3_smarts", {})
        row["fr_t3_solved"] = t3.get("solved", False)
        row["fr_t3_score"] = t3.get("score", 0)
        row["fr_t3_time"] = t3.get("time_s", 0)

        # AiZynthFinder
        aizf = aizf_by_smi.get(smi, {})
        row["aizf_solved"] = aizf.get("solved", False)
        row["aizf_n_routes"] = aizf.get("n_routes", 0)
        row["aizf_time"] = aizf.get("time_s", 0)

        best = aizf.get("best_route", {}) or {}
        row["aizf_steps"] = best.get("n_steps", 0)
        row["aizf_bbs"] = best.get("n_bbs", 0)

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_comparison_summary(rows: list[dict]) -> dict:
    """Compute aggregate comparison statistics."""
    n = len(rows)
    if n == 0:
        return {}

    def solve_rate(key):
        return sum(1 for r in rows if r.get(key, False)) / n

    def mean_val(key, condition_key=None):
        vals = []
        for r in rows:
            if condition_key and not r.get(condition_key, False):
                continue
            v = r.get(key, 0)
            if v is not None and v > 0:
                vals.append(v)
        return sum(vals) / len(vals) if vals else 0

    def median_val(key, condition_key=None):
        vals = []
        for r in rows:
            if condition_key and not r.get(condition_key, False):
                continue
            v = r.get(key, 0)
            if v is not None:
                vals.append(v)
        vals.sort()
        return vals[len(vals)//2] if vals else 0

    summary = {
        "n_molecules": n,
        "methods": {},
    }

    # Define methods and their column mappings
    methods = {
        "FR Tier 1 (binary)": {
            "solve_key": "fr_t1b_solved",
            "time_key": "fr_t1b_time",
        },
        "FR Tier 1 (continuous)": {
            "score_key": "fr_t1c_score",
            "time_key": "fr_t1c_time",
        },
        "FR Tier 1 (DAG)": {
            "solve_key": "fr_t1d_solved",
            "score_key": "fr_t1d_score",
            "time_key": "fr_t1d_time",
            "steps_key": "fr_t1d_steps",
            "lls_key": "fr_t1d_lls",
            "bbs_key": "fr_t1d_bbs",
        },
        "FR Tier 2 (SMARTS-validated)": {
            "solve_key": "fr_t2_solved",
            "score_key": "fr_t2_score",
            "time_key": "fr_t2_time",
            "steps_key": "fr_t2_steps",
            "lls_key": "fr_t2_lls",
            "bbs_key": "fr_t2_bbs",
        },
        "FR Tier 3 (SMARTS retro)": {
            "solve_key": "fr_t3_solved",
            "score_key": "fr_t3_score",
            "time_key": "fr_t3_time",
        },
        "AiZynthFinder": {
            "solve_key": "aizf_solved",
            "time_key": "aizf_time",
            "steps_key": "aizf_steps",
            "bbs_key": "aizf_bbs",
        },
    }

    for name, keys in methods.items():
        m = {}

        if "solve_key" in keys:
            m["solve_rate"] = round(solve_rate(keys["solve_key"]), 4)
            m["solved"] = sum(1 for r in rows if r.get(keys["solve_key"], False))

        if "score_key" in keys:
            m["mean_score"] = round(mean_val(keys["score_key"]), 4)

        if "time_key" in keys:
            times = [r[keys["time_key"]] for r in rows if keys["time_key"] in r]
            if times:
                m["mean_time_s"] = round(sum(times) / len(times), 4)
                m["median_time_s"] = round(sorted(times)[len(times)//2], 4)
                m["total_time_s"] = round(sum(times), 2)
                m["speedup_vs_aizf"] = None  # computed below

        solve_key = keys.get("solve_key")
        if "steps_key" in keys and solve_key:
            m["mean_steps"] = round(mean_val(keys["steps_key"], solve_key), 2)
        if "lls_key" in keys and solve_key:
            m["mean_lls"] = round(mean_val(keys["lls_key"], solve_key), 2)
        if "bbs_key" in keys and solve_key:
            m["mean_bbs"] = round(mean_val(keys["bbs_key"], solve_key), 2)

        summary["methods"][name] = m

    # Compute speedup relative to AiZynthFinder
    aizf_time = summary["methods"].get("AiZynthFinder", {}).get("mean_time_s", 0)
    if aizf_time > 0:
        for name, m in summary["methods"].items():
            if "mean_time_s" in m and m["mean_time_s"] > 0:
                m["speedup_vs_aizf"] = round(aizf_time / m["mean_time_s"], 1)

    return summary


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_comparison_table(summary: dict) -> None:
    """Print a formatted comparison table to console."""
    methods = summary.get("methods", {})
    n = summary.get("n_molecules", 0)

    header = (f"{'Method':<30s}  {'Solve%':>7s}  {'Score':>7s}  "
              f"{'Time/mol':>9s}  {'Speedup':>8s}  "
              f"{'Steps':>6s}  {'LLS':>5s}  {'BBs':>5s}")
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(f"BENCHMARK COMPARISON (n={n})")
    print(sep)
    print(header)
    print(sep)

    for name, m in methods.items():
        solve = f"{m.get('solve_rate', 0):.1%}" if "solve_rate" in m else "—"
        score = f"{m.get('mean_score', 0):.3f}" if "mean_score" in m else "—"
        time_s = f"{m.get('mean_time_s', 0):.3f}s" if "mean_time_s" in m else "—"
        speedup = f"{m.get('speedup_vs_aizf', 0):.0f}×" if m.get("speedup_vs_aizf") else "1×"
        steps = f"{m.get('mean_steps', 0):.1f}" if "mean_steps" in m else "—"
        lls = f"{m.get('mean_lls', 0):.1f}" if "mean_lls" in m else "—"
        bbs = f"{m.get('mean_bbs', 0):.1f}" if "mean_bbs" in m else "—"

        print(f"{name:<30s}  {solve:>7s}  {score:>7s}  "
              f"{time_s:>9s}  {speedup:>8s}  "
              f"{steps:>6s}  {lls:>5s}  {bbs:>5s}")

    print(sep)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write per-molecule comparison as CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {path}")


def write_latex_table(summary: dict, path: Path) -> None:
    """Write a LaTeX-formatted comparison table."""
    methods = summary.get("methods", {})
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Retrosynthesis benchmark comparison on USPTO-190}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Solve\% & Score & Time/mol (s) & Speedup & Steps & BBs \\",
        r"\midrule",
    ]

    for name, m in methods.items():
        # Shorten name for LaTeX
        short = name.replace("FR ", "").replace("(", "").replace(")", "")
        solve = f"{m.get('solve_rate', 0):.1%}"
        score = f"{m.get('mean_score', 0):.3f}" if "mean_score" in m else "—"
        time_s = f"{m.get('mean_time_s', 0):.3f}"
        speedup = f"{m.get('speedup_vs_aizf', 0):.0f}$\\times$" if m.get("speedup_vs_aizf") else "1$\\times$"
        steps = f"{m.get('mean_steps', 0):.1f}" if "mean_steps" in m else "—"
        bbs = f"{m.get('mean_bbs', 0):.1f}" if "mean_bbs" in m else "—"

        lines.append(f"  {short} & {solve} & {score} & {time_s} & {speedup} & {steps} & {bbs} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_solve_rate_comparison(summary: dict, output_dir: Path) -> None:
    """Bar chart comparing solve rates."""
    if not HAS_MPL:
        return

    methods = summary.get("methods", {})
    names = list(methods.keys())
    rates = [methods[n].get("solve_rate", 0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB", "#FF9800"]
    bars = ax.bar(range(len(names)), rates, color=colors[:len(names)], edgecolor="white")

    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Retrosynthesis Solve Rate — USPTO-190", fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = output_dir / "solve_rate_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_speed_comparison(summary: dict, output_dir: Path) -> None:
    """Bar chart comparing speed (time per molecule)."""
    if not HAS_MPL:
        return

    methods = summary.get("methods", {})
    names = list(methods.keys())
    times = [methods[n].get("mean_time_s", 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9", "#FF5722"]
    bars = ax.bar(range(len(names)), times, color=colors[:len(names)], edgecolor="white")

    ax.set_ylabel("Time per molecule (seconds)", fontsize=12)
    ax.set_title("Retrosynthesis Speed — USPTO-190", fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yscale("log")

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f"{t:.3f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = output_dir / "speed_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_solve_vs_time_scatter(rows: list[dict], output_dir: Path) -> None:
    """Scatter: FR Tier 2 time vs AiZynthFinder time, colored by solve."""
    if not HAS_MPL:
        return

    fr_times = []
    aizf_times = []
    colors = []

    for r in rows:
        fr_t = r.get("fr_t2_time", r.get("fr_t1d_time", 0))
        aizf_t = r.get("aizf_time", 0)
        if fr_t > 0 and aizf_t > 0:
            fr_times.append(fr_t)
            aizf_times.append(aizf_t)
            fr_solved = r.get("fr_t2_solved", r.get("fr_t1d_solved", False))
            aizf_solved = r.get("aizf_solved", False)
            if fr_solved and aizf_solved:
                colors.append("#4CAF50")  # both
            elif fr_solved:
                colors.append("#2196F3")  # FR only
            elif aizf_solved:
                colors.append("#FF9800")  # AiZF only
            else:
                colors.append("#9E9E9E")  # neither

    if not fr_times:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(aizf_times, fr_times, c=colors, alpha=0.6, edgecolors="white", s=40)

    # Diagonal reference
    max_t = max(max(fr_times), max(aizf_times)) * 1.1
    ax.plot([0.001, max_t], [0.001, max_t], "k--", alpha=0.3, label="equal time")

    ax.set_xlabel("AiZynthFinder time (s)", fontsize=12)
    ax.set_ylabel("FragmentRetro time (s)", fontsize=12)
    ax.set_title("Per-molecule speed comparison", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#4CAF50", label="Both solved"),
        Patch(facecolor="#2196F3", label="FR only"),
        Patch(facecolor="#FF9800", label="AiZF only"),
        Patch(facecolor="#9E9E9E", label="Neither"),
    ]
    ax.legend(handles=legend_items, loc="upper left")

    plt.tight_layout()
    path = output_dir / "time_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_route_quality(rows: list[dict], output_dir: Path) -> None:
    """Box plot comparing route quality (steps, BBs) for solved molecules."""
    if not HAS_MPL:
        return

    # Collect data
    data = {
        "FR Tier 1 DAG": {"steps": [], "bbs": []},
        "FR Tier 2": {"steps": [], "bbs": []},
        "AiZynthFinder": {"steps": [], "bbs": []},
    }

    for r in rows:
        if r.get("fr_t1d_solved") and r.get("fr_t1d_steps", 0) > 0:
            data["FR Tier 1 DAG"]["steps"].append(r["fr_t1d_steps"])
            data["FR Tier 1 DAG"]["bbs"].append(r["fr_t1d_bbs"])
        if r.get("fr_t2_solved") and r.get("fr_t2_steps", 0) > 0:
            data["FR Tier 2"]["steps"].append(r["fr_t2_steps"])
            data["FR Tier 2"]["bbs"].append(r["fr_t2_bbs"])
        if r.get("aizf_solved") and r.get("aizf_steps", 0) > 0:
            data["AiZynthFinder"]["steps"].append(r["aizf_steps"])
            data["AiZynthFinder"]["bbs"].append(r["aizf_bbs"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Steps box plot
    labels = []
    steps_data = []
    for name, d in data.items():
        if d["steps"]:
            labels.append(name)
            steps_data.append(d["steps"])

    if steps_data:
        bp1 = ax1.boxplot(steps_data, labels=labels, patch_artist=True)
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        for patch, color in zip(bp1["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel("Number of Steps")
        ax1.set_title("Synthesis Steps (solved molecules)")

    # BBs box plot
    labels2 = []
    bbs_data = []
    for name, d in data.items():
        if d["bbs"]:
            labels2.append(name)
            bbs_data.append(d["bbs"])

    if bbs_data:
        bp2 = ax2.boxplot(bbs_data, labels=labels2, patch_artist=True)
        for patch, color in zip(bp2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel("Number of Building Blocks")
        ax2.set_title("Building Blocks (solved molecules)")

    plt.tight_layout()
    path = output_dir / "route_quality_boxplot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


def plot_solve_by_complexity(rows: list[dict], output_dir: Path) -> None:
    """Line plot: solve rate vs molecule size (heavy atom bins)."""
    if not HAS_MPL:
        return

    # Bin by heavy atom count
    bins = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 100)]
    bin_labels = ["≤15", "16-20", "21-25", "26-30", "31-40", ">40"]

    methods = {
        "FR Tier 1": "fr_t1d_solved",
        "FR Tier 2": "fr_t2_solved",
        "FR Tier 3": "fr_t3_solved",
        "AiZynthFinder": "aizf_solved",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {
        "FR Tier 1": "#2196F3",
        "FR Tier 2": "#4CAF50",
        "FR Tier 3": "#9C27B0",
        "AiZynthFinder": "#FF9800",
    }

    for method_name, key in methods.items():
        rates = []
        for lo, hi in bins:
            in_bin = [r for r in rows if lo < r.get("num_heavy_atoms", 0) <= hi]
            if in_bin:
                solved = sum(1 for r in in_bin if r.get(key, False))
                rates.append(solved / len(in_bin) * 100)
            else:
                rates.append(0)

        ax.plot(range(len(bins)), rates, "o-", label=method_name,
                color=colors_map[method_name], linewidth=2, markersize=6)

    ax.set_xlabel("Heavy Atom Count", fontsize=12)
    ax.set_ylabel("Solve Rate (%)", fontsize=12)
    ax.set_title("Solve Rate vs Molecular Complexity", fontsize=14)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "solve_by_complexity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot: {path}")


# ---------------------------------------------------------------------------
# Venn overlap analysis
# ---------------------------------------------------------------------------

def compute_overlap(rows: list[dict]) -> dict:
    """Compute solve overlap between methods."""
    fr_only = set()
    aizf_only = set()
    both = set()
    neither = set()

    for r in rows:
        smi = r["smiles"]
        # Use best FR tier available
        fr_solved = (r.get("fr_t2_solved", False) or
                     r.get("fr_t1d_solved", False) or
                     r.get("fr_t3_solved", False))
        aizf_solved = r.get("aizf_solved", False)

        if fr_solved and aizf_solved:
            both.add(smi)
        elif fr_solved:
            fr_only.add(smi)
        elif aizf_solved:
            aizf_only.add(smi)
        else:
            neither.add(smi)

    return {
        "both": len(both),
        "fr_only": len(fr_only),
        "aizf_only": len(aizf_only),
        "neither": len(neither),
        "total": len(rows),
        "fr_total": len(both) + len(fr_only),
        "aizf_total": len(both) + len(aizf_only),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to benchmark_config.json")
    parser.add_argument("--fr-results", type=str, default=None,
                        help="Path to results_fragmentretro.json")
    parser.add_argument("--aizf-results", type=str, default=None,
                        help="Path to results_aizynthfinder.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config_dir = Path(config["targets_file"]).parent

    fr_path = args.fr_results or str(config_dir / "results_fragmentretro.json")
    aizf_path = args.aizf_results or str(config_dir / "results_aizynthfinder.json")
    output_dir = Path(args.output_dir or str(config_dir / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results ...")
    fr_data = load_results(fr_path)
    aizf_data = load_results(aizf_path)

    print(f"  FragmentRetro: {len(fr_data.get('results', []))} molecules")
    print(f"  AiZynthFinder: {len(aizf_data.get('results', []))} molecules")

    # Build comparison table
    rows = build_comparison_table(fr_data, aizf_data)
    print(f"  Matched: {len(rows)} molecules for comparison")

    # Summary statistics
    summary = compute_comparison_summary(rows)

    # Overlap analysis
    overlap = compute_overlap(rows)
    summary["overlap"] = overlap

    # Console output
    print_comparison_table(summary)

    print(f"\nSolve overlap:")
    print(f"  Both solve:     {overlap['both']}")
    print(f"  FR only:        {overlap['fr_only']}")
    print(f"  AiZF only:      {overlap['aizf_only']}")
    print(f"  Neither:        {overlap['neither']}")

    # Write outputs
    write_csv(rows, output_dir / "comparison_per_molecule.csv")
    write_latex_table(summary, output_dir / "comparison_table.tex")

    # Summary JSON
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    # Plots
    print(f"\nGenerating plots ...")
    plot_solve_rate_comparison(summary, output_dir)
    plot_speed_comparison(summary, output_dir)
    plot_solve_vs_time_scatter(rows, output_dir)
    plot_route_quality(rows, output_dir)
    plot_solve_by_complexity(rows, output_dir)

    print(f"\nAnalysis complete → {output_dir}/")


if __name__ == "__main__":
    main()
