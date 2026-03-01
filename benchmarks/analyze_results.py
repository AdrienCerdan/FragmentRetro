#!/usr/bin/env python
"""Compare FragmentRetro vs AiZynthFinder benchmark results.

Reads results JSON files produced by run_fragmentretro.py and
run_aizynthfinder.py, then generates:
  - Summary comparison table (printed + saved as TSV)
  - Per-molecule comparison CSV
  - Visualisation plots (if matplotlib is available)

Usage:
    python analyze_results.py --datadir benchmarks/data

    # Custom input files
    python analyze_results.py \
        --fragmentretro results_fragmentretro.json \
        --aizynthfinder results_aizynthfinder.json \
        --outdir benchmarks/analysis

    # Skip plots (no matplotlib needed)
    python analyze_results.py --datadir benchmarks/data --no-plots
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fragmentretro(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_aizynthfinder(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def safe_mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def safe_median(lst: list[float]) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def safe_std(lst: list[float]) -> float:
    if len(lst) < 2:
        return 0.0
    m = safe_mean(lst)
    return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5


def percentile(lst: list[float], p: float) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def extract_per_molecule(data: dict, tool: str) -> dict[str, dict]:
    """Extract per-molecule results keyed by SMILES.

    Returns {smiles: {metric: value, ...}} for the best available tier.
    """
    molecules = {}

    if tool == "fragmentretro":
        # Prefer higher tiers (more informative)
        for tier_key in ["tier_3", "tier_2", "tier_1"]:
            tier_data = data.get("results", {}).get(tier_key, {})
            for mol in tier_data.get("molecules", []):
                smi = mol["smiles"]
                if smi not in molecules or mol.get("solved", False):
                    molecules[smi] = {
                        "tool": "fragmentretro",
                        "tier": mol.get("tier", 0),
                        "solved": mol.get("solved", False),
                        "score": mol.get("score", 0.0),
                        "wall_time_s": mol.get("wall_time_s", 0.0),
                        "total_steps": mol.get("total_steps", 0),
                        "lls": mol.get("longest_linear_sequence", 0),
                        "num_bbs": mol.get("num_building_blocks", 0),
                        "timed_out": mol.get("timed_out", False),
                        "error": mol.get("error"),
                    }
    elif tool == "aizynthfinder":
        for mol in data.get("molecules", []):
            smi = mol["smiles"]
            molecules[smi] = {
                "tool": "aizynthfinder",
                "solved": mol.get("solved", False),
                "score": mol.get("score", 0.0),
                "wall_time_s": mol.get("wall_time_s", 0.0),
                "total_steps": mol.get("total_steps", 0),
                "lls": mol.get("longest_linear_sequence", 0),
                "num_bbs": mol.get("num_building_blocks", 0),
                "timed_out": mol.get("timed_out", False),
                "error": mol.get("error"),
                "num_routes_found": mol.get("num_routes_found", 0),
            }

    return molecules


def compute_summary_row(label: str, molecules: dict[str, dict]) -> dict:
    """Compute aggregate stats from per-molecule results."""
    n = len(molecules)
    if n == 0:
        return {"label": label, "n": 0}

    solved_list = [m for m in molecules.values() if m.get("solved")]
    times = [m["wall_time_s"] for m in molecules.values()]
    solved_times = [m["wall_time_s"] for m in solved_list]
    steps = [m["total_steps"] for m in solved_list if m["total_steps"] > 0]
    lls = [m["lls"] for m in solved_list if m["lls"] > 0]
    bbs = [m["num_bbs"] for m in solved_list if m["num_bbs"] > 0]
    timeouts = sum(1 for m in molecules.values() if m.get("timed_out"))
    errors = sum(1 for m in molecules.values() if m.get("error"))

    return {
        "label": label,
        "n": n,
        "solved": len(solved_list),
        "solved_pct": round(100 * len(solved_list) / n, 1),
        "timeouts": timeouts,
        "errors": errors,
        "mean_time": round(safe_mean(times), 3),
        "median_time": round(safe_median(times), 3),
        "p95_time": round(percentile(times, 95), 3),
        "total_time": round(sum(times), 1),
        "mean_steps": round(safe_mean(steps), 2),
        "median_steps": round(safe_median(steps), 2),
        "mean_lls": round(safe_mean(lls), 2),
        "mean_bbs": round(safe_mean(bbs), 2),
        "mean_solved_time": round(safe_mean(solved_times), 3),
    }


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table."""
    cols = [
        ("Tool/Tier", "label", "{}"),
        ("N", "n", "{}"),
        ("Solved", "solved", "{}"),
        ("Rate%", "solved_pct", "{:.1f}"),
        ("T.out", "timeouts", "{}"),
        ("Err", "errors", "{}"),
        ("Mean t(s)", "mean_time", "{:.3f}"),
        ("Med t(s)", "median_time", "{:.3f}"),
        ("P95 t(s)", "p95_time", "{:.3f}"),
        ("Total(s)", "total_time", "{:.1f}"),
        ("Avg steps", "mean_steps", "{:.2f}"),
        ("Avg LLS", "mean_lls", "{:.2f}"),
        ("Avg BBs", "mean_bbs", "{:.2f}"),
    ]

    widths = {name: max(len(name), 9) for name, _, _ in cols}
    header = " | ".join(f"{name:>{widths[name]}}" for name, _, _ in cols)
    sep = "-+-".join("-" * widths[name] for name, _, _ in cols)

    print()
    print(header)
    print(sep)

    for row in rows:
        cells = []
        for name, key, fmt in cols:
            val = row.get(key, "")
            try:
                cell = fmt.format(val)
            except (ValueError, TypeError):
                cell = str(val)
            cells.append(f"{cell:>{widths[name]}}")
        print(" | ".join(cells))

    print()


def write_per_molecule_csv(
    fr_mols: dict[str, dict],
    az_mols: dict[str, dict],
    output_path: Path,
) -> None:
    """Write per-molecule comparison CSV."""
    all_smiles = sorted(set(fr_mols.keys()) | set(az_mols.keys()))

    fields = [
        "smiles",
        "fr_solved", "fr_score", "fr_time_s", "fr_steps", "fr_lls", "fr_bbs", "fr_tier",
        "az_solved", "az_score", "az_time_s", "az_steps", "az_lls", "az_bbs", "az_routes",
        "both_solved", "fr_only", "az_only", "neither",
        "speedup",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for smi in all_smiles:
            fr = fr_mols.get(smi, {})
            az = az_mols.get(smi, {})

            fr_solved = fr.get("solved", False)
            az_solved = az.get("solved", False)
            fr_time = fr.get("wall_time_s", 0)
            az_time = az.get("wall_time_s", 0)

            speedup = (az_time / fr_time) if fr_time > 0 and az_time > 0 else 0

            writer.writerow({
                "smiles": smi,
                "fr_solved": int(fr_solved),
                "fr_score": round(fr.get("score", 0), 4),
                "fr_time_s": round(fr_time, 4),
                "fr_steps": fr.get("total_steps", 0),
                "fr_lls": fr.get("lls", 0),
                "fr_bbs": fr.get("num_bbs", 0),
                "fr_tier": fr.get("tier", ""),
                "az_solved": int(az_solved),
                "az_score": round(az.get("score", 0), 4),
                "az_time_s": round(az_time, 4),
                "az_steps": az.get("total_steps", 0),
                "az_lls": az.get("lls", 0),
                "az_bbs": az.get("num_bbs", 0),
                "az_routes": az.get("num_routes_found", 0),
                "both_solved": int(fr_solved and az_solved),
                "fr_only": int(fr_solved and not az_solved),
                "az_only": int(az_solved and not fr_solved),
                "neither": int(not fr_solved and not az_solved),
                "speedup": round(speedup, 2),
            })

    print(f"  Per-molecule comparison written to {output_path}")


def write_summary_tsv(rows: list[dict], output_path: Path) -> None:
    """Write summary table as TSV."""
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Summary table written to {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(
    fr_mols: dict[str, dict],
    az_mols: dict[str, dict],
    summary_rows: list[dict],
    outdir: Path,
) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    common = sorted(set(fr_mols.keys()) & set(az_mols.keys()))
    if not common:
        print("  No common molecules for plotting")
        return

    # --- 1. Solve rate bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["label"] for r in summary_rows]
    solved_pcts = [r.get("solved_pct", 0) for r in summary_rows]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(labels)]
    bars = ax.bar(labels, solved_pcts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, solved_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("USPTO 190 — Solve Rate Comparison")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "solve_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved solve_rate.png")

    # --- 2. Time distribution box plot ---
    time_data = []
    time_labels = []
    for row in summary_rows:
        label = row["label"]
        if "FR" in label or "frag" in label.lower():
            key = "fragmentretro"
            mols = fr_mols
        else:
            key = "aizynthfinder"
            mols = az_mols
        times = [m["wall_time_s"] for m in mols.values() if not m.get("timed_out")]
        if times:
            time_data.append(times)
            time_labels.append(label)

    if time_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        bp = ax.boxplot(time_data, labels=time_labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], colors[:len(time_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Wall Time (s)")
        ax.set_title("Time Distribution per Molecule")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "time_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved time_distribution.png")

    # --- 3. Scatter: FR time vs AiZynth time ---
    fr_times = [fr_mols[s]["wall_time_s"] for s in common]
    az_times = [az_mols[s]["wall_time_s"] for s in common]

    fig, ax = plt.subplots(figsize=(7, 7))
    fr_solved_mask = [fr_mols[s].get("solved", False) and az_mols[s].get("solved", False) for s in common]
    fr_only_mask = [fr_mols[s].get("solved", False) and not az_mols[s].get("solved", False) for s in common]
    az_only_mask = [not fr_mols[s].get("solved", False) and az_mols[s].get("solved", False) for s in common]
    neither_mask = [not fr_mols[s].get("solved", False) and not az_mols[s].get("solved", False) for s in common]

    for mask, label, color, marker in [
        (fr_solved_mask, "Both solved", "#4CAF50", "o"),
        (fr_only_mask, "FR only", "#2196F3", "^"),
        (az_only_mask, "AiZynth only", "#F44336", "v"),
        (neither_mask, "Neither", "#9E9E9E", "x"),
    ]:
        xs = [fr_times[i] for i in range(len(common)) if mask[i]]
        ys = [az_times[i] for i in range(len(common)) if mask[i]]
        if xs:
            ax.scatter(xs, ys, label=label, color=color, marker=marker, alpha=0.6, s=30)

    lims = [min(min(fr_times), min(az_times)) * 0.5,
            max(max(fr_times), max(az_times)) * 2]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FragmentRetro Time (s)")
    ax.set_ylabel("AiZynthFinder Time (s)")
    ax.set_title("Per-Molecule Time Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "time_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved time_scatter.png")

    # --- 4. Route quality: steps distribution ---
    fr_steps = [fr_mols[s]["total_steps"] for s in common if fr_mols[s].get("solved")]
    az_steps = [az_mols[s]["total_steps"] for s in common if az_mols[s].get("solved")]

    if fr_steps or az_steps:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = range(0, max(max(fr_steps, default=5), max(az_steps, default=5)) + 2)
        if fr_steps:
            ax.hist(fr_steps, bins=bins, alpha=0.6, label="FragmentRetro", color="#2196F3")
        if az_steps:
            ax.hist(az_steps, bins=bins, alpha=0.6, label="AiZynthFinder", color="#F44336")
        ax.set_xlabel("Number of Synthesis Steps")
        ax.set_ylabel("Count (solved molecules)")
        ax.set_title("Route Length Distribution (Solved Molecules)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "steps_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Saved steps_distribution.png")

    # --- 5. Venn-style solve overlap ---
    both = sum(1 for s in common if fr_mols[s].get("solved") and az_mols[s].get("solved"))
    fr_only = sum(1 for s in common if fr_mols[s].get("solved") and not az_mols[s].get("solved"))
    az_only = sum(1 for s in common if not fr_mols[s].get("solved") and az_mols[s].get("solved"))
    neither = sum(1 for s in common if not fr_mols[s].get("solved") and not az_mols[s].get("solved"))

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Both", "FR only", "AiZynth only", "Neither"]
    counts = [both, fr_only, az_only, neither]
    colors_v = ["#4CAF50", "#2196F3", "#F44336", "#9E9E9E"]
    bars = ax.barh(categories, counts, color=colors_v)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), ha="left", va="center", fontsize=11)
    ax.set_xlabel("Number of molecules")
    ax.set_title("Solve Overlap (Common Targets)")
    fig.tight_layout()
    fig.savefig(outdir / "solve_overlap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved solve_overlap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare FragmentRetro vs AiZynthFinder benchmark results"
    )
    parser.add_argument(
        "--datadir", type=Path, default=Path("benchmarks/data"),
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--fragmentretro", type=Path, default=None,
        help="Path to results_fragmentretro.json"
    )
    parser.add_argument(
        "--aizynthfinder", type=Path, default=None,
        help="Path to results_aizynthfinder.json"
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory for analysis (default: datadir/analysis)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation"
    )
    args = parser.parse_args()

    fr_path = args.fragmentretro or (args.datadir / "results_fragmentretro.json")
    az_path = args.aizynthfinder or (args.datadir / "results_aizynthfinder.json")
    outdir = args.outdir or (args.datadir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    has_fr = fr_path.exists()
    has_az = az_path.exists()

    if not has_fr and not has_az:
        print("Error: no result files found")
        print(f"  Expected: {fr_path}")
        print(f"       and: {az_path}")
        sys.exit(1)

    fr_data = load_fragmentretro(fr_path) if has_fr else None
    az_data = load_aizynthfinder(az_path) if has_az else None

    # Extract per-molecule data
    fr_mols: dict[str, dict] = {}
    az_mols: dict[str, dict] = {}
    summary_rows: list[dict] = []

    if fr_data:
        # One row per tier
        for tier_key in ["tier_1", "tier_2", "tier_3"]:
            tier_data = fr_data.get("results", {}).get(tier_key)
            if not tier_data:
                continue
            tier_num = int(tier_key.split("_")[1])
            tier_mols = {}
            for mol in tier_data.get("molecules", []):
                tier_mols[mol["smiles"]] = {
                    "tool": "fragmentretro",
                    "tier": tier_num,
                    "solved": mol.get("solved", False),
                    "score": mol.get("score", 0.0),
                    "wall_time_s": mol.get("wall_time_s", 0.0),
                    "total_steps": mol.get("total_steps", 0),
                    "lls": mol.get("longest_linear_sequence", 0),
                    "num_bbs": mol.get("num_building_blocks", 0),
                    "timed_out": mol.get("timed_out", False),
                    "error": mol.get("error"),
                }
            summary_rows.append(compute_summary_row(f"FR Tier {tier_num}", tier_mols))

        # Best FR result per molecule (prefer solved, then highest tier)
        for tier_key in ["tier_1", "tier_2", "tier_3"]:
            tier_data = fr_data.get("results", {}).get(tier_key)
            if not tier_data:
                continue
            tier_num = int(tier_key.split("_")[1])
            for mol in tier_data.get("molecules", []):
                smi = mol["smiles"]
                new_entry = {
                    "tool": "fragmentretro",
                    "tier": tier_num,
                    "solved": mol.get("solved", False),
                    "score": mol.get("score", 0.0),
                    "wall_time_s": mol.get("wall_time_s", 0.0),
                    "total_steps": mol.get("total_steps", 0),
                    "lls": mol.get("longest_linear_sequence", 0),
                    "num_bbs": mol.get("num_building_blocks", 0),
                    "timed_out": mol.get("timed_out", False),
                    "error": mol.get("error"),
                }
                existing = fr_mols.get(smi)
                if existing is None:
                    fr_mols[smi] = new_entry
                elif new_entry["solved"] and not existing["solved"]:
                    fr_mols[smi] = new_entry

    if az_data:
        az_mols = extract_per_molecule(az_data, "aizynthfinder")
        summary_rows.append(compute_summary_row("AiZynthFinder", az_mols))

    # Print comparison table
    print("\n" + "=" * 100)
    print("  BENCHMARK COMPARISON: FragmentRetro vs AiZynthFinder")
    print("=" * 100)
    print_comparison_table(summary_rows)

    # Overlap analysis
    if has_fr and has_az:
        common = set(fr_mols.keys()) & set(az_mols.keys())
        both = sum(1 for s in common if fr_mols[s].get("solved") and az_mols[s].get("solved"))
        fr_only = sum(1 for s in common if fr_mols[s].get("solved") and not az_mols[s].get("solved"))
        az_only = sum(1 for s in common if not fr_mols[s].get("solved") and az_mols[s].get("solved"))
        neither = sum(1 for s in common if not fr_mols[s].get("solved") and not az_mols[s].get("solved"))

        print(f"  Overlap analysis ({len(common)} common targets):")
        print(f"    Both solved:       {both}")
        print(f"    FR only:           {fr_only}")
        print(f"    AiZynthFinder only:{az_only}")
        print(f"    Neither:           {neither}")

        # Speed comparison on commonly solved
        common_solved = [s for s in common
                         if fr_mols[s].get("solved") and az_mols[s].get("solved")]
        if common_solved:
            fr_times = [fr_mols[s]["wall_time_s"] for s in common_solved]
            az_times = [az_mols[s]["wall_time_s"] for s in common_solved]
            mean_speedup = safe_mean(az_times) / safe_mean(fr_times) if safe_mean(fr_times) > 0 else 0
            median_speedup = safe_median(az_times) / safe_median(fr_times) if safe_median(fr_times) > 0 else 0
            print(f"\n  Speed on {len(common_solved)} commonly-solved molecules:")
            print(f"    FR  mean={safe_mean(fr_times):.3f}s  median={safe_median(fr_times):.3f}s")
            print(f"    AZ  mean={safe_mean(az_times):.3f}s  median={safe_median(az_times):.3f}s")
            print(f"    Speedup (AZ/FR): mean={mean_speedup:.1f}x  median={median_speedup:.1f}x")
        print()

    # Write outputs
    print("Saving analysis outputs...")
    write_summary_tsv(summary_rows, outdir / "summary.tsv")

    if has_fr and has_az:
        write_per_molecule_csv(fr_mols, az_mols, outdir / "per_molecule.csv")

    if not args.no_plots and has_fr and has_az:
        print("Generating plots...")
        generate_plots(fr_mols, az_mols, summary_rows, outdir)

    print(f"\n  All outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
