#!/usr/bin/env python
"""Prepare benchmark data: USPTO 190 targets and eMolecules stock.

This script prepares the two inputs needed for benchmarking:
  1. Target molecules (USPTO 190 test set)
  2. Building block stock (eMolecules ~200K)

It validates SMILES, canonicalises them, and precomputes FragmentRetro's
mol_properties.json from the stock file.

Usage:
    # With your own files
    python prepare_data.py \
        --targets /path/to/uspto190.smi \
        --stock /path/to/emolecules_200k.smi \
        --outdir benchmarks/data

    # Generate mol_properties only (targets already prepared)
    python prepare_data.py \
        --stock /path/to/emolecules_200k.smi \
        --outdir benchmarks/data \
        --skip-targets

File formats accepted:
    Targets: .smi / .csv / .txt — one SMILES per line (optional header)
    Stock:   .smi / .csv / .txt — one SMILES per line (optional header)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


def canonicalise(smiles: str) -> str | None:
    """Canonicalise a SMILES string. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def read_smiles_file(path: Path) -> list[str]:
    """Read SMILES from a file, handling .smi, .csv, and .txt.

    Supports:
      - One SMILES per line (with optional trailing name/id)
      - CSV with 'smiles' or 'SMILES' column
      - Lines starting with # are skipped
    """
    raw = path.read_text().strip().splitlines()

    # Skip empty / comment lines
    lines = [l.strip() for l in raw if l.strip() and not l.startswith("#")]
    if not lines:
        return []

    # Detect CSV with header
    first = lines[0]
    if "," in first and any(h in first.lower() for h in ("smiles", "smi")):
        reader = csv.DictReader(lines)
        col = None
        for c in reader.fieldnames or []:
            if c.lower() in ("smiles", "smi", "canonical_smiles"):
                col = c
                break
        if col is None:
            print(f"  Warning: CSV detected but no SMILES column found in {path}")
            return []
        return [row[col] for row in reader if row.get(col)]

    # Tab/space separated — take first column, skip header-like lines
    smiles_list = []
    for line in lines:
        parts = line.split()
        smi = parts[0]
        # Skip likely header rows
        if smi.lower() in ("smiles", "smi", "canonical_smiles", "id", "name"):
            continue
        smiles_list.append(smi)

    return smiles_list


def validate_and_canonicalise(
    smiles_list: list[str], label: str
) -> list[str]:
    """Validate and canonicalise a list of SMILES. Reports stats."""
    valid = []
    invalid = 0
    seen = set()

    for smi in tqdm(smiles_list, desc=f"Validating {label}"):
        canon = canonicalise(smi)
        if canon is None:
            invalid += 1
            continue
        if canon in seen:
            continue
        seen.add(canon)
        valid.append(canon)

    print(f"  {label}: {len(smiles_list)} input → {len(valid)} valid unique "
          f"({invalid} invalid, {len(smiles_list) - len(valid) - invalid} duplicates)")
    return valid


def precompute_mol_properties(
    smiles_list: list[str],
    output_path: Path,
    fp_size: int = 2048,
) -> None:
    """Precompute FragmentRetro mol_properties.json from stock SMILES.

    This is the format CompoundFilter expects for BB lookup.
    """
    from rdkit.Chem import rdMolDescriptors

    results = []
    errors = 0

    for smi in tqdm(smiles_list, desc="Precomputing mol properties"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                errors += 1
                continue
            mol.UpdatePropertyCache()
            Chem.GetSymmSSSR(mol)

            pfp = list(Chem.rdmolops.PatternFingerprint(mol, fpSize=fp_size).GetOnBits())
            results.append({
                "cano_smiles": smi,
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "pfp": pfp,
            })
        except Exception:
            errors += 1

    with open(output_path, "w") as f:
        json.dump(results, f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {len(results)} BB entries to {output_path} ({size_mb:.1f} MB)")
    if errors:
        print(f"  ({errors} entries failed)")


def create_aizynthfinder_stock(smiles_list: list[str], output_path: Path) -> None:
    """Write stock in AiZynthFinder-compatible format (one SMILES per line)."""
    output_path.write_text("\n".join(smiles_list) + "\n")
    print(f"  Wrote {len(smiles_list)} BB SMILES to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare benchmark data for FragmentRetro vs AiZynthFinder"
    )
    parser.add_argument(
        "--targets", type=Path, required=True,
        help="Path to USPTO 190 targets file (one SMILES per line)"
    )
    parser.add_argument(
        "--stock", type=Path, required=True,
        help="Path to eMolecules stock file (one SMILES per line)"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("benchmarks/data"),
        help="Output directory (default: benchmarks/data)"
    )
    parser.add_argument(
        "--fp-size", type=int, default=2048,
        help="Fingerprint size for mol_properties (default: 2048)"
    )
    parser.add_argument(
        "--skip-targets", action="store_true",
        help="Skip target validation (use existing targets.smi)"
    )
    parser.add_argument(
        "--skip-mol-properties", action="store_true",
        help="Skip mol_properties precomputation"
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Targets ---
    targets_out = args.outdir / "targets.smi"
    if not args.skip_targets:
        print("\n=== Preparing target molecules ===")
        if not args.targets.exists():
            print(f"Error: targets file not found: {args.targets}")
            sys.exit(1)

        raw_targets = read_smiles_file(args.targets)
        targets = validate_and_canonicalise(raw_targets, "targets")

        targets_out.write_text("\n".join(targets) + "\n")
        print(f"  Wrote {len(targets)} targets to {targets_out}")
    else:
        if targets_out.exists():
            targets = targets_out.read_text().strip().splitlines()
            print(f"  Using existing {targets_out} ({len(targets)} targets)")
        else:
            print(f"Error: --skip-targets but {targets_out} does not exist")
            sys.exit(1)

    # --- Stock ---
    print("\n=== Preparing building block stock ===")
    if not args.stock.exists():
        print(f"Error: stock file not found: {args.stock}")
        sys.exit(1)

    raw_stock = read_smiles_file(args.stock)
    stock = validate_and_canonicalise(raw_stock, "stock")

    # AiZynthFinder stock format
    stock_smi_out = args.outdir / "stock.smi"
    create_aizynthfinder_stock(stock, stock_smi_out)

    # FragmentRetro mol_properties
    mol_props_out = args.outdir / "mol_properties.json"
    if not args.skip_mol_properties:
        print("\n=== Precomputing FragmentRetro mol_properties ===")
        t0 = time.time()
        precompute_mol_properties(stock, mol_props_out, fp_size=args.fp_size)
        elapsed = time.time() - t0
        print(f"  Precomputation took {elapsed:.1f}s")
    else:
        print(f"  Skipping mol_properties (use existing {mol_props_out})")

    # --- Summary ---
    print("\n=== Prepared data ===")
    print(f"  Targets:          {targets_out} ({len(targets)} molecules)")
    print(f"  Stock (SMILES):   {stock_smi_out} ({len(stock)} BBs)")
    print(f"  Stock (FR props): {mol_props_out}")
    print()
    print("Next steps:")
    print(f"  python benchmarks/run_fragmentretro.py --outdir benchmarks/data")
    print(f"  python benchmarks/run_aizynthfinder.py --outdir benchmarks/data")
    print(f"  python benchmarks/analyze_results.py   --outdir benchmarks/data")


if __name__ == "__main__":
    main()
