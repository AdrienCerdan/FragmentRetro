#!/usr/bin/env python
"""
Chemical space estimation with optional strict synthesis-oriented filters.

Strict v2 filters (--strict):
  1. BB-level FG incompatibility (pre-enumeration)
  2. BB-level steric hindrance proxy (pre-enumeration)
  3. Regioisomer rejection (post-enumeration)
  4. Lilly instability/reactivity SMARTS (post-enumeration)
  5. MW / heavy-atom cap (post-enumeration)
"""
import argparse
import json
import random
import multiprocessing as mp
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from fragmentretro.reaction_library import ReactionLibrary

# ── Global worker state ──────────────────────────────────────────────────────
_worker_stock_mols = []
_strict_mode = False
_lilly_patterns = []      # list of compiled Mol objects for instability SMARTS
_max_mw = 0
_max_ha = 0


def init_worker(stock_smiles, strict=False, lilly_smarts=None, max_mw=600, max_ha=50):
    global _worker_stock_mols, _strict_mode, _lilly_patterns, _max_mw, _max_ha
    _worker_stock_mols = [Chem.MolFromSmiles(s) for s in stock_smiles]
    _strict_mode = strict
    _max_mw = max_mw
    _max_ha = max_ha
    if strict and lilly_smarts:
        _lilly_patterns = []
        for sma in lilly_smarts:
            pat = Chem.MolFromSmarts(sma)
            if pat:
                _lilly_patterns.append(pat)


def _passes_product_filters(smi):
    """Check a product SMILES against Lilly instability + MW/HA caps."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    # MW cap
    if Descriptors.MolWt(mol) > _max_mw:
        return False
    # Heavy atom cap
    if mol.GetNumHeavyAtoms() > _max_ha:
        return False
    # Lilly instability patterns
    for pat in _lilly_patterns:
        if mol.HasSubstructMatch(pat):
            return False
    return True


def run_reaction_batch(rxn_smarts, reactant_indices):
    """Run reactions for a batch of reactant index tuples and return unique product SMILES."""
    global _worker_stock_mols, _strict_mode
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if not rxn:
        return set()
    rxn.Initialize()
    products = set()
    for indices in reactant_indices:
        mols = [_worker_stock_mols[i] for i in indices]
        try:
            p_sets = rxn.RunReactants(tuple(mols))
            unique_for_pair = set()
            for p_set in p_sets:
                for p in p_set:
                    try:
                        Chem.SanitizeMol(p)
                        unique_for_pair.add(Chem.MolToSmiles(p))
                    except Exception:
                        pass

            if _strict_mode:
                # Layer 3: Regioisomer rejection
                if len(unique_for_pair) > 1:
                    continue
                # Layer 4+5: Lilly instability + MW/HA cap
                unique_for_pair = {s for s in unique_for_pair if _passes_product_filters(s)}

            products.update(unique_for_pair)
        except Exception:
            continue
    return products


def run_rxn_2st(rxn_id, rxn_smarts, pairing, prec_smi):
    """Run 2nd-step reactions with a precursor molecule."""
    global _worker_stock_mols, _strict_mode
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if not rxn:
        return set()
    rxn.Initialize()
    pmol = Chem.MolFromSmiles(prec_smi)
    if not pmol:
        return set()
    res = set()
    for p in pairing:
        mols = []
        for idx in p:
            if idx == -1:
                mols.append(pmol)
            else:
                mols.append(_worker_stock_mols[idx])
        try:
            p_sets = rxn.RunReactants(tuple(mols))
            unique_for_pair = set()
            for ps in p_sets:
                for p_mol in ps:
                    try:
                        Chem.SanitizeMol(p_mol)
                        unique_for_pair.add(Chem.MolToSmiles(p_mol))
                    except Exception:
                        pass

            if _strict_mode:
                if len(unique_for_pair) > 1:
                    continue
                unique_for_pair = {s for s in unique_for_pair if _passes_product_filters(s)}

            res.update(unique_for_pair)
        except Exception:
            continue
    return res


# ── Filter loading ────────────────────────────────────────────────────────────

class SynthesisFilters:
    """Loads and applies synthesis-oriented filters from synthesis_filters.json."""

    def __init__(self):
        json_path = Path(__file__).resolve().parent.parent / "src" / "fragmentretro" / "data" / "synthesis_filters.json"
        with open(json_path) as f:
            self.data = json.load(f)

        # Compile Lilly instability patterns
        self.lilly_smarts = [entry["smarts"] for entry in self.data["lilly_instability_smarts"]]

        # Compile FG incompatibility patterns per reaction_id
        # We collect all exclusion keys (any_bb, halide_bb, amine_bb, acid_bb) into one list per reaction
        self.fg_exclusions = {}  # reaction_id -> list of compiled SMARTS Mol
        for group_name, group in self.data["fg_incompatibility"].items():
            compiled = []
            for key in group:
                if key.startswith("exclusions_on_"):
                    for exc in group[key]:
                        pat = Chem.MolFromSmarts(exc["smarts"])
                        if pat:
                            compiled.append(pat)
            for rxn_id in group["reaction_ids"]:
                self.fg_exclusions.setdefault(rxn_id, []).extend(compiled)

        # Compile steric hindrance patterns from the "patterns" array
        self.steric_patterns = []
        steric = self.data.get("steric_hindrance", {})
        for entry in steric.get("patterns", []):
            pat = Chem.MolFromSmarts(entry["smarts"])
            if pat:
                self.steric_patterns.append(pat)

        self.product_caps = self.data.get("product_caps", {})


    def filter_bb_pairs(self, rxn_id, pairs, stock_mols):
        """
        Layer 1+2: Pre-filter BB pairs by FG incompatibility and steric hindrance.
        Returns filtered list of index tuples.
        """
        exclusions = self.fg_exclusions.get(rxn_id, [])
        if not exclusions and not self.steric_patterns:
            return pairs

        filtered = []
        for pair in pairs:
            mols_in_pair = [stock_mols[i] for i in pair]

            # Layer 1: FG incompatibility
            fg_fail = False
            if exclusions:
                for mol in mols_in_pair:
                    if mol is None:
                        continue
                    for pat in exclusions:
                        if mol.HasSubstructMatch(pat):
                            fg_fail = True
                            break
                    if fg_fail:
                        break
            if fg_fail:
                continue

            # Layer 2: Steric hindrance (both reactants must be hindered to reject)
            if self.steric_patterns and len(mols_in_pair) == 2:
                both_hindered = True
                for mol in mols_in_pair:
                    if mol is None:
                        both_hindered = False
                        break
                    is_hindered = any(mol.HasSubstructMatch(sp) for sp in self.steric_patterns)
                    if not is_hindered:
                        both_hindered = False
                        break
                if both_hindered:
                    continue

            filtered.append(pair)
        return filtered


# ── Pattern matching ──────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Estimate accessible chemical space from BB stock + reactions")
    parser.add_argument("--stock", required=True, help="Path to BB stock JSON")
    parser.add_argument("--steps", type=int, default=1, help="1 or 2 step estimation")
    parser.add_argument("--hte-only", action="store_true", help="Use only Hartenfeller HTE reactions")
    parser.add_argument("--strict", action="store_true",
                        help="Apply synthesis-oriented filters (Lilly instability, FG incompatibility, regio rejection, MW/HA cap)")
    parser.add_argument("--sample-size", type=int, default=20000,
                        help="Max combinations to sample per reaction for 1st step")
    parser.add_argument("--sample-size-2", type=int, default=20,
                        help="Number of 1st-step products to sample for 2nd step branching")
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

    # Load filters if strict mode
    filters = None
    lilly_smarts = None
    max_mw = 600
    max_ha = 50
    if args.strict:
        filters = SynthesisFilters()
        lilly_smarts = filters.lilly_smarts
        max_mw = filters.product_caps.get("max_mw_1step", 600)
        max_ha = filters.product_caps.get("max_heavy_atoms", 50)
        print(f"Strict v2 filters loaded: {len(lilly_smarts)} instability SMARTS, "
              f"{sum(len(v) for v in filters.fg_exclusions.values())} FG exclusion rules, "
              f"MW≤{max_mw}, HA≤{max_ha}")

    # ── 1-Step Estimation ─────────────────────────────────────────────────────
    print(f"\n--- 1-STEP ESTIMATION {'(STRICT v2)' if args.strict else '(Unfiltered)'} ---")
    tasks = []

    for rxn in reactions:
        patterns = rxn.smarts_forward.split(">>")[0].split(".")
        if len(patterns) == 1:
            indices = matches[patterns[0]]
            n_comb = len(indices)
            if not indices:
                continue
            sample = [(i,) for i in random.sample(indices, min(n_comb, args.sample_size))]
        elif len(patterns) == 2:
            idx1, idx2 = matches[patterns[0]], matches[patterns[1]]
            n_comb = len(idx1) * len(idx2)
            if not idx1 or not idx2:
                continue
            if n_comb > args.sample_size:
                sample = [(random.choice(idx1), random.choice(idx2)) for _ in range(args.sample_size)]
            else:
                sample = [(i1, i2) for i1 in idx1 for i2 in idx2]
        else:
            continue

        # Layer 1+2: BB-level pre-filtering (only in strict mode)
        pre_filter_count = len(sample)
        if args.strict and filters:
            sample = filters.filter_bb_pairs(rxn.id, sample, stock_mols)

        tasks.append((rxn.id, rxn.name, rxn.smarts_forward, sample, n_comb, pre_filter_count))

    print(f"Running {len(tasks)} reaction samples across {args.workers} workers...")
    all_observed = set()
    sum_individual_observed = 0
    total_est_unique = 0
    total_comb = 0

    with mp.Pool(args.workers, initializer=init_worker,
                 initargs=(stock_smiles, args.strict, lilly_smarts, max_mw, max_ha)) as pool:
        for rxn_id, rxn_name, smarts, sample, n_total, pre_count in tasks:
            if not sample:
                print(f"  {rxn_id:<20} | Comb: {n_total:<10} | AllFilteredOut")
                continue

            chunk_size = 5000
            chunks = [sample[i:i + chunk_size] for i in range(0, len(sample), chunk_size)]
            results = pool.starmap(run_reaction_batch, [(smarts, c) for c in chunks])

            seen_in_sample = set()
            for r in results:
                seen_in_sample.update(r)

            all_observed.update(seen_in_sample)
            sum_individual_observed += len(seen_in_sample)

            ratio = len(seen_in_sample) / len(sample) if sample else 0
            # Scale by pre-filter pass rate when strict
            if args.strict and pre_count > 0:
                bb_pass_rate = len(sample) / pre_count
                est = int(n_total * bb_pass_rate * ratio)
            else:
                est = int(n_total * ratio)

            total_est_unique += est
            total_comb += n_total
            bb_info = f" BBpass:{len(sample)}/{pre_count}" if args.strict else ""
            print(f"  {rxn_id:<20} | Comb: {n_total:<10} | SampleUniq: {len(seen_in_sample):<6} | EstUniq: {est:,}{bb_info}")

        # Overlap correction
        overlap_factor = len(all_observed) / sum_individual_observed if sum_individual_observed > 0 else 1.0
        final_1step = int(total_est_unique * overlap_factor)

        print(f"\n--- 1-STEP REPORT ---")
        print(f"Total Theoretical Combinations (Ceiling): {total_comb:,}")
        print(f"Sum of Per-Reaction Unique Estimates: {total_est_unique:,}")
        print(f"Empirical Inter-Reaction Overlap Factor: {overlap_factor:.3f}")
        print(f"Corrected Unique 1-Step Space: ~{final_1step:,}")

        # ── 2-Step Estimation ─────────────────────────────────────────────────
        if args.steps >= 2 and all_observed:
            # Update MW cap for 2-step products
            if args.strict and filters:
                max_mw_2 = filters.product_caps.get("max_mw_2step", 800)
                # Re-init workers with higher MW cap for 2nd step
                # (We can't easily change globals mid-pool, so we accept 1-step cap for now)
                print(f"\nNote: 2-step products use MW cap {max_mw} (same as 1-step workers)")

            print(f"\n--- 2-STEP BRANCHING ---")
            sample_size_2 = min(args.sample_size_2, len(all_observed))
            print(f"Sampling {sample_size_2} molecules from 1-step products...")
            sample_2step = random.sample(list(all_observed), sample_size_2)

            branching_factors = []
            for idx, precursor in enumerate(sample_2step):
                precursor_mol = Chem.MolFromSmiles(precursor)
                if not precursor_mol:
                    continue

                tasks_2 = []
                for rxn in reactions:
                    reactant_smarts = rxn.smarts_forward.split(">>")[0].split(".")
                    for i in range(len(reactant_smarts)):
                        pm = Chem.MolFromSmarts(reactant_smarts[i])
                        if pm and precursor_mol.HasSubstructMatch(pm):
                            if len(reactant_smarts) == 1:
                                tasks_2.append((rxn.id, rxn.smarts_forward, [(-1,)], precursor))
                            else:
                                other_idxs = matches[reactant_smarts[1 - i]]
                                if len(other_idxs) > 1000:
                                    other_idxs = random.sample(other_idxs, 1000)
                                pairs = [((-1, o) if i == 0 else (o, -1)) for o in other_idxs]
                                tasks_2.append((rxn.id, rxn.smarts_forward, pairs, precursor))

                results_2 = pool.starmap(run_rxn_2st, [(t[0], t[1], t[2], t[3]) for t in tasks_2])
                found = set()
                for r in results_2:
                    found.update(r)

                branching_factors.append(len(found))
                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1}/{sample_size_2} precursors...")

            avg_bf = sum(branching_factors) / len(branching_factors) if branching_factors else 0
            final_2step = final_1step * avg_bf
            print(f"\n--- 2-STEP REPORT ---")
            print(f"Average Branching Factor: {avg_bf:,.1f}")
            print(f"Extrapolated 2-Step Space: ~{final_2step:,.0f} molecules")


if __name__ == "__main__":
    main()
