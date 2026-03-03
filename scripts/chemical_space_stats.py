#!/usr/bin/env python
import argparse
import json
import collections
from rdkit import Chem
from fragmentretro.reaction_library import ReactionLibrary, Reaction

def main():
    parser = argparse.ArgumentParser(description="Analyze fruitfulness of reactions and building blocks.")
    parser.add_argument("--stock", type=str, required=True, help="Path to json or SMILES file")
    parser.add_argument("--hte-only", action="store_true", help="Only use HTE-compatible reactions")
    args = parser.parse_args()

    # 1. Load Stock
    print(f"Loading stock from {args.stock}...")
    if args.stock.endswith(".json"):
        with open(args.stock) as f:
            data = json.load(f)
            stock_smiles = [item["cano_smiles"] for item in data] if isinstance(data, list) else list(data.keys())
    else:
        with open(args.stock) as f:
            stock_smiles = [line.strip().split()[0] for line in f if line.strip()]

    print(f"Parsing {len(stock_smiles)} molecules for matching...")
    stock_mols = [Chem.MolFromSmiles(s) for s in stock_smiles]
    # Filter out invalid
    valid_idx = [i for i, m in enumerate(stock_mols) if m is not None]
    stock_mols = [stock_mols[i] for i in valid_idx]
    stock_smiles = [stock_smiles[i] for i in valid_idx]

    # 2. Load Reactions
    lib = ReactionLibrary.default()
    reactions = [r for r in lib.reactions if r.source == "hartenfeller"] if args.hte_only else lib.reactions

    # 3. Match patterns
    all_patterns = set()
    for rxn in reactions:
        reactant_smarts = rxn.smarts_forward.split(">>")[0].split(".")
        all_patterns.update(reactant_smarts)

    print(f"Pre-filtering stock for {len(all_patterns)} unique reactant patterns...")
    pattern_mols = {ps: Chem.MolFromSmarts(ps) for ps in all_patterns if Chem.MolFromSmarts(ps)}
    
    matches = {ps: [] for ps in all_patterns}
    bb_participation = collections.defaultdict(int) # idx -> count of patterns matched
    
    for i, mol in enumerate(stock_mols):
        matched_any = False
        for ps, pm in pattern_mols.items():
            if mol.HasSubstructMatch(pm):
                matches[ps].append(i)
                bb_participation[i] += 1
                matched_any = True

    # 4. Reaction Fruitfulness (Combinations)
    rxn_stats = []
    for rxn in reactions:
        reactant_smarts = rxn.smarts_forward.split(">>")[0].split(".")
        if len(reactant_smarts) == 1:
            n_comb = len(matches[reactant_smarts[0]])
        elif len(reactant_smarts) == 2:
            n_comb = len(matches[reactant_smarts[0]]) * len(matches[reactant_smarts[1]])
        else: n_comb = 0
        rxn_stats.append((rxn.name, rxn.id, n_comb))

    rxn_stats.sort(key=lambda x: x[2], reverse=True)

    # 5. BB Fruitfulness (Participation)
    bb_stats = []
    for idx, count in bb_participation.items():
        bb_stats.append((stock_smiles[idx], count))
    bb_stats.sort(key=lambda x: x[1], reverse=True)

    print("\n--- TOP 10 FRUITFUL REACTIONS (By Potential Combinations) ---")
    print(f"{'Reaction Name':<30} | {'ID':<15} | {'Combinations':<15}")
    print("-" * 65)
    for name, rid, n in rxn_stats[:15]:
        print(f"{name:<30} | {rid:<15} | {n:,}")

    print("\n--- TOP 10 VERSATILE BUILDING BLOCKS (By Reaction Patterns Matched) ---")
    print(f"{'SMILES':<50} | {'Patterns':<10}")
    print("-" * 65)
    for smiles, count in bb_stats[:10]:
        print(f"{smiles[:48]:<50} | {count:<10}")

    print("\n--- GENERAL STATS ---")
    print(f"Total BBs: {len(stock_smiles)}")
    print(f"BBs matching at least one pattern: {len(bb_participation)} ({len(bb_participation)/len(stock_smiles):.1%})")
    print(f"Reactions with at least one combination: {len([x for x in rxn_stats if x[2] > 0])}/{len(reactions)}")

if __name__ == "__main__":
    main()
