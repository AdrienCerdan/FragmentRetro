#!/usr/bin/env python
import argparse
import json
import random
import multiprocessing as mp
import collections
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import FilterCatalog
from fragmentretro.reaction_library import ReactionLibrary

# Global storage for worker processes to avoid re-parsing
_worker_stock_mols = []
_filter_catalog = None
_strict_mode = False

def init_worker(stock_smiles, strict=False):
    global _worker_stock_mols, _filter_catalog, _strict_mode
    _worker_stock_mols = [Chem.MolFromSmiles(s) for s in stock_smiles]
    _strict_mode = strict
    if strict:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _filter_catalog = FilterCatalog.FilterCatalog(params)

def run_reaction_batch(rxn_smarts, reactant_indices):
    global _worker_stock_mols, _filter_catalog, _strict_mode
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if not rxn: return set()
    rxn.Initialize()
    products = set()
    for indices in reactant_indices:
        mols = [_worker_stock_mols[i] for i in indices]
        try:
            p_sets = rxn.RunReactants(tuple(mols))
            unique_smi_for_pair = set()
            for p_set in p_sets:
                for p in p_set:
                    try:
                        Chem.SanitizeMol(p)
                        smi = Chem.MolToSmiles(p)
                        unique_smi_for_pair.add(smi)
                    except: pass
            
            if _strict_mode:
                if len(unique_smi_for_pair) > 1:
                    continue
                valid_smiles = set()
                for smi in unique_smi_for_pair:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and not _filter_catalog.HasMatch(mol):
                        valid_smiles.add(smi)
                unique_smi_for_pair = valid_smiles
            products.update(unique_smi_for_pair)
        except: continue
    return products

def run_rxn_2st(rxn_id, rxn_smarts, pairing, prec_smi):
    global _worker_stock_mols, _filter_catalog, _strict_mode
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if not rxn: return set()
    rxn.Initialize()
    pmol = Chem.MolFromSmiles(prec_smi)
    if not pmol: return set()
    res = set()
    for p in pairing:
        mols = []
        for idx in p:
            if idx == -1: mols.append(pmol)
            else: mols.append(_worker_stock_mols[idx])
        try:
            p_sets = rxn.RunReactants(tuple(mols))
            unique_smi_for_pair = set()
            for ps in p_sets:
                for p_mol in ps:
                    try:
                        Chem.SanitizeMol(p_mol)
                        unique_smi_for_pair.add(Chem.MolToSmiles(p_mol))
                    except: pass
            
            if _strict_mode:
                if len(unique_smi_for_pair) > 1:
                    continue
                valid_smiles = set()
                for smi in unique_smi_for_pair:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and not _filter_catalog.HasMatch(mol):
                        valid_smiles.add(smi)
                unique_smi_for_pair = valid_smiles
            res.update(unique_smi_for_pair)
        except: continue
    return res

def get_matching_bbs(reactions, stock_mols):
    all_patterns = set()
    for rxn in reactions:
        reactant_smarts = rxn.smarts_forward.split(">>")[0].split(".")
        all_patterns.update(reactant_smarts)
    pattern_mols = {ps: Chem.MolFromSmarts(ps) for ps in all_patterns if Chem.MolFromSmarts(ps)}
    matches = {ps: [] for ps in all_patterns}
    for i, mol in enumerate(stock_mols):
        for ps, pm in pattern_mols.items():
            if mol.HasSubstructMatch(pm):
                matches[ps].append(i)
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--hte-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Apply PAINS/BRENK/NIH filters and regioselectivity constraints")
    parser.add_argument("--sample-size", type=int, default=20000, help="Max combinations to sample per reaction for 1st step")
    parser.add_argument("--sample-size-2", type=int, default=20, help="Number of 1st-step products to sample for 2nd step branching")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Load building blocks
    with open(args.stock) as f:
        data = json.load(f)
        stock_smiles = [item["cano_smiles"] for item in data] if isinstance(data, list) else list(data.keys())
    stock_mols = [Chem.MolFromSmiles(s) for s in stock_smiles]
    stock_mols = [m for m in stock_mols if m]
    print(f"Loaded {len(stock_mols)} building blocks.")

    # Load reactions & match patterns
    lib = ReactionLibrary.default()
    reactions = [r for r in lib.reactions if r.source == "hartenfeller"] if args.hte_only else lib.reactions
    matches = get_matching_bbs(reactions, stock_mols)

    print("\n--- 1-STEP ESTIMATION (Empirical Sampling) ---")
    tasks = []
    
    for rxn in reactions:
        patterns = rxn.smarts_forward.split(">>")[0].split(".")
        if len(patterns) == 1:
            indices = matches[patterns[0]]
            n_comb = len(indices)
            if not indices: continue
            sample = [(i,) for i in (random.sample(indices, min(n_comb, args.sample_size)))]
        elif len(patterns) == 2:
            idx1, idx2 = matches[patterns[0]], matches[patterns[1]]
            n_comb = len(idx1) * len(idx2)
            if not idx1 or not idx2: continue
            if n_comb > args.sample_size:
                sample = [(random.choice(idx1), random.choice(idx2)) for _ in range(args.sample_size)]
            else:
                sample = [(i1, i2) for i1 in idx1 for i2 in idx2]
        else: continue
        tasks.append((rxn.id, rxn.name, rxn.smarts_forward, sample, n_comb))

    print(f"Running {len(tasks)} reaction samples across {args.workers} workers...")
    all_observed = set()
    sum_individual_observed = 0
    total_est_unique = 0
    total_comb = 0

    with mp.Pool(args.workers, initializer=init_worker, initargs=(stock_smiles, args.strict)) as pool:
        for rxn_id, rxn_name, smarts, sample, n_total in tasks:
            chunk_size = 5000
            chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]
            results = pool.starmap(run_reaction_batch, [(smarts, c) for c in chunks])
            
            seen_in_sample = set()
            for r in results: seen_in_sample.update(r)
            
            all_observed.update(seen_in_sample)
            sum_individual_observed += len(seen_in_sample)
            
            ratio = len(seen_in_sample) / len(sample) if sample else 0
            est = int(n_total * ratio)
            total_est_unique += est
            total_comb += n_total
            print(f"  {rxn_id:<15} | Comb: {n_total:<10} | SampleUniq: {len(seen_in_sample):<6} | EstUniq: {est:,}")

        # Overlap Correction
        overlap_factor = len(all_observed) / sum_individual_observed if sum_individual_observed > 0 else 1.0
        final_1step = int(total_est_unique * overlap_factor)
        
        print("\n--- 1-STEP STATISTICAL REPORT ---")
        print(f"Total Theoretical Combinations (Ceiling): {total_comb:,}")
        print(f"Sum of Individual Reaction Unique Estimates: {total_est_unique:,}")
        print(f"Empirical Inter-Reaction Overlap Factor: {overlap_factor:.3f} (Lower = more redundancy)")
        print(f"Corrected Unique 1-Step Space: ~{final_1step:,}")

        if args.steps >= 2 and all_observed:
            print(f"\n--- 2-STEP BRANCHING ---")
            sample_size_2 = min(args.sample_size_2, len(all_observed))
            print(f"Sampling {sample_size_2} molecules from 1-step products to calculate branching factor...")
            sample_2step = random.sample(list(all_observed), sample_size_2)
            
            branching_factors = []
            for idx, precursor in enumerate(sample_2step):
                precursor_mol = Chem.MolFromSmiles(precursor)
                if not precursor_mol: continue
                
                tasks_2 = []
                for rxn in reactions:
                    reactant_smarts = rxn.smarts_forward.split(">>")[0].split(".")
                    for i in range(len(reactant_smarts)):
                        pm = Chem.MolFromSmarts(reactant_smarts[i])
                        if precursor_mol.HasSubstructMatch(pm):
                            if len(reactant_smarts) == 1:
                                tasks_2.append((rxn.id, rxn.smarts_forward, [(-1,)]))
                            else:
                                other_idxs = matches[reactant_smarts[1 - i]]
                                # Sub-sample if necessary to keep 2nd step fast during estimation
                                if len(other_idxs) > 1000: other_idxs = random.sample(other_idxs, 1000)
                                pairs = [((-1, o_idx) if i == 0 else (o_idx, -1)) for o_idx in other_idxs]
                                tasks_2.append((rxn.id, rxn.smarts_forward, pairs))

                results_2 = pool.starmap(run_rxn_2st, [(t[0], t[1], t[2], precursor) for t in tasks_2])
                found = set()
                for r in results_2: found.update(r)
                
                # Correct for sub-sampling in 2nd step (if we sampled 1000 pairs out of length(other_idxs))
                # To be fully accurate, we calculate proportion, but for branching factor, raw generated is a good lower bound
                branching_factors.append(len(found))
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx+1}/{sample_size_2} precursors...")

            avg_bf = sum(branching_factors) / len(branching_factors) if branching_factors else 0
            final_2step = final_1step * avg_bf
            print(f"\n--- 2-STEP STATISTICAL REPORT ---")
            print(f"Average New Products per 1-Step Precursor (Branching Factor): {avg_bf:,.1f}")
            print(f"Extrapolated 2-Step Space: ~{final_2step:,.0f} molecules")

if __name__ == "__main__":
    main()
