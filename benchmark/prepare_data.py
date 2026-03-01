#!/usr/bin/env python
"""Prepare benchmark data: USPTO-190 targets and building block stock.

Downloads or converts:
  1. USPTO-190 target molecules (190 benchmark SMILES)
  2. Building block stock -> FragmentRetro mol_properties.json format
  3. Building block stock -> AiZynthFinder stock format

Usage:
    python prepare_data.py \
        --stock-file /path/to/emolecules_200k.smi \
        --output-dir ./benchmark_data

    python prepare_data.py \
        --stock-file /path/to/stock.smi \
        --targets-file /path/to/targets.smi \
        --output-dir ./benchmark_data
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# AiZynthFinder's standard USPTO-190 benchmark set
USPTO_190_URL = (
    "https://raw.githubusercontent.com/MolecularAI/aizynthfinder/"
    "master/tests/data/retrosynthesis/targets_190.smi"
)


def download_uspto_190(output_path: Path) -> list[str]:
    """Download USPTO-190 targets from AiZynthFinder repo."""
    print(f"Downloading USPTO-190 from {USPTO_190_URL} ...")
    try:
        response = urllib.request.urlopen(USPTO_190_URL, timeout=30)
        content = response.read().decode("utf-8")
        smiles_list = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                smi = line.split("\t")[0].split()[0]
                if smi:
                    smiles_list.append(smi)
        print(f"  Downloaded {len(smiles_list)} target molecules")
    except Exception as e:
        print(f"  Download failed ({e})")
        print("  Please provide --targets-file manually")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for smi in smiles_list:
            f.write(f"{smi}\n")
    return smiles_list


def load_targets(path: Path) -> list[str]:
    """Load target SMILES from a file (one per line, or .smi/.csv)."""
    smiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split("\t")[0].split(",")[0].split()[0]
            if smi:
                smiles.append(smi)
    return smiles


def load_stock_smiles(path: Path) -> list[str]:
    """Load building block SMILES from .smi, .csv, or .txt."""
    ext = path.suffix.lower()
    smiles = []

    if ext == ".csv":
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            smi_col = 0
            if header:
                for i, h in enumerate(header):
                    if h.lower() in ("smiles", "smi", "canonical_smiles", "mol"):
                        smi_col = i
                        break
            for row in reader:
                if len(row) > smi_col:
                    smi = row[smi_col].strip()
                    if smi and not smi.startswith("#"):
                        smiles.append(smi)
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                smi = line.split("\t")[0].split()[0]
                if smi:
                    smiles.append(smi)
    return smiles


def prepare_fragmentretro_stock(stock_smiles, output_path, fp_size=2048):
    """Convert stock SMILES to FragmentRetro mol_properties.json."""
    from fragmentretro.utils.filter_compound import precompute_properties

    print(f"Precomputing FragmentRetro properties for {len(stock_smiles)} BBs ...")
    print(f"  Fingerprint size: {fp_size}")
    print(f"  Output: {output_path}")

    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precompute_properties(stock_smiles, output_path, fpSize=fp_size)
    print(f"  Done in {time.time() - t0:.1f}s")


def prepare_aizynthfinder_stock(stock_smiles, output_path):
    """Write stock as plain-text SMILES + optional HDF5."""
    print(f"Writing AiZynthFinder stock ({len(stock_smiles)} BBs) ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for smi in stock_smiles:
            f.write(f"{smi}\n")

    hdf5_path = output_path.with_suffix(".hdf5")
    try:
        import pandas as pd
        df = pd.DataFrame({"SMILES": stock_smiles})
        df.to_hdf(hdf5_path, key="table", mode="w")
        print(f"  HDF5 stock: {hdf5_path}")
    except ImportError:
        print("  (pandas not available, skipping HDF5)")

    print(f"  Text stock: {output_path}")


def write_benchmark_config(output_dir, targets_path, stock_path_fr,
                           stock_path_aizf, n_targets, n_stock):
    """Write benchmark_config.json linking all paths."""
    config = {
        "targets_file": str(targets_path),
        "n_targets": n_targets,
        "fragmentretro_stock": str(stock_path_fr),
        "aizynthfinder_stock": str(stock_path_aizf),
        "aizynthfinder_stock_hdf5": str(stock_path_aizf.with_suffix(".hdf5")),
        "n_stock_bbs": n_stock,
        "fp_size": 2048,
    }
    config_path = output_dir / "benchmark_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nBenchmark config: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data")
    parser.add_argument("--stock-file", type=str, required=True,
                        help="Path to building block stock (SMILES file)")
    parser.add_argument("--targets-file", type=str, default=None,
                        help="Path to target molecules (default: download USPTO-190)")
    parser.add_argument("--output-dir", type=str, default="./benchmark_data",
                        help="Output directory")
    parser.add_argument("--fp-size", type=int, default=2048,
                        help="Fingerprint size for FragmentRetro")
    parser.add_argument("--max-stock", type=int, default=None,
                        help="Limit stock to first N BBs (for quick tests)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Targets ---
    if args.targets_file:
        targets = load_targets(Path(args.targets_file))
        out_targets = output_dir / "targets.smi"
        with open(out_targets, "w") as f:
            for smi in targets:
                f.write(f"{smi}\n")
    else:
        out_targets = output_dir / "targets.smi"
        targets = download_uspto_190(out_targets)

    print(f"Targets: {len(targets)} molecules -> {out_targets}")

    # --- Stock ---
    stock_smiles = load_stock_smiles(Path(args.stock_file))
    print(f"\nLoaded {len(stock_smiles)} BBs from {args.stock_file}")

    if args.max_stock and len(stock_smiles) > args.max_stock:
        stock_smiles = stock_smiles[:args.max_stock]
        print(f"  Truncated to {len(stock_smiles)} BBs")

    stock_smiles = list(dict.fromkeys(stock_smiles))
    print(f"  Unique BBs: {len(stock_smiles)}")

    fr_stock_path = output_dir / "mol_properties.json"
    prepare_fragmentretro_stock(stock_smiles, fr_stock_path, fp_size=args.fp_size)

    aizf_stock_path = output_dir / "aizynthfinder_stock.txt"
    prepare_aizynthfinder_stock(stock_smiles, aizf_stock_path)

    write_benchmark_config(output_dir, out_targets, fr_stock_path,
                           aizf_stock_path, len(targets), len(stock_smiles))

    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"  Targets:          {len(targets)}")
    print(f"  Building blocks:  {len(stock_smiles)}")
    print(f"  Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  python benchmark/run_fragmentretro.py --config {output_dir / 'benchmark_config.json'}")
    print(f"  python benchmark/run_aizynthfinder.py --config {output_dir / 'benchmark_config.json'}")
    print(f"  python benchmark/analyze_results.py --config {output_dir / 'benchmark_config.json'}")


if __name__ == "__main__":
    main()
