#!/usr/bin/env python
"""Benchmark FragmentRetro across all tiers.

Runs the full target set through:
  - Tier 1:  BRICS-only binary / continuous / DAG
  - Tier 2:  BRICS + SMARTS validation
  - Tier 3:  Standalone SMARTS retrosynthesis

Usage:
    # Serial (default)
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json \
        --tiers 1 2 3 --timeout 120

    # Parallel with 4 workers
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json \
        --workers 4

    # Auto-detect CPU count
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json -j 0

    # Quick test
    python run_fragmentretro.py \
        --config ./benchmark_data/benchmark_config.json \
        --max-targets 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

from fragmentretro.scoring import (
    compute_all_brics_tiers,
    load_compound_filter,
)
from fragmentretro.reaction_library import ReactionLibrary
from fragmentretro.utils.logging_config import logger, setup_logging

setup_logging()


# Module-level globals for multiprocessing workers.
# Initialized once per worker via _init_worker() to avoid
# re-serializing the (large) CompoundFilter/ReactionLibrary
# for every task.
_worker_cf = None
_worker_lib = None
_worker_tiers = None
_worker_timeout = None
_worker_max_depth = None
_worker_max_depth = None
_worker_max_nodes = None
_worker_reference_routes = None

def extract_reference_leaves(node):
    if node.get("type") == "mol" and node.get("in_stock", False):
        return {node["smiles"]}
    leaves = set()
    for child in node.get("children", []):
        leaves.update(extract_reference_leaves(child))
    return leaves

def load_reference_routes(path):
    print(f"Loading reference routes from {path} ...")
    with open(path, 'r') as f:
        ref_data = json.load(f)
    reference_routes = {}
    for item in ref_data:
        smiles = item["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            can_smiles = Chem.MolToSmiles(mol)
            leaves = extract_reference_leaves(item)
            can_leaves = set()
            for l in leaves:
                lmol = Chem.MolFromSmiles(l)
                if lmol:
                    can_leaves.add(Chem.MolToSmiles(lmol))
                else:
                    can_leaves.add(l)
            reference_routes[can_smiles] = list(can_leaves)
    return reference_routes

def get_retro_node_leaves(node):
    if node is None:
        return set()
    if getattr(node, "is_leaf", False):
        bbs = getattr(node, "bb_smiles", None)
        if bbs:
            can_bbs = set()
            for bb in bbs:
                mol = Chem.MolFromSmiles(bb)
                if mol:
                    can_bbs.add(Chem.MolToSmiles(mol))
                else:
                    can_bbs.add(bb)
            return can_bbs
        # Fallback to fragment smiles
        mol = Chem.MolFromSmiles(node.smiles)
        if mol:
            return {Chem.MolToSmiles(mol)}
        return {node.smiles}
    leaves = set()
    for c in getattr(node, "children", []):
        leaves.update(get_retro_node_leaves(c))
    return leaves

def _smiles_to_fp(smi, radius=2, nbits=2048):
    """Convert SMILES to Morgan fingerprint, or None on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def compute_route_similarity(ref_leaves, pred_leaves):
    """Compute similarity metrics between reference and predicted BB sets.

    Args:
        ref_leaves: set of canonical SMILES (reference building blocks).
        pred_leaves: set of canonical SMILES (predicted building blocks).

    Returns:
        dict with jaccard, recall, precision, max_tanimoto, n_ref, n_pred, n_common.
    """
    if not ref_leaves or not pred_leaves:
        return {
            "jaccard": 0.0, "recall": 0.0, "precision": 0.0,
            "max_tanimoto": 0.0, "n_ref": len(ref_leaves) if ref_leaves else 0,
            "n_pred": len(pred_leaves) if pred_leaves else 0, "n_common": 0,
        }

    common = ref_leaves & pred_leaves
    union = ref_leaves | pred_leaves
    jaccard = len(common) / len(union) if union else 0.0
    recall = len(common) / len(ref_leaves)
    precision = len(common) / len(pred_leaves)

    # Tanimoto fingerprint similarity: for each predicted BB, find the max
    # Tanimoto to any reference BB. Average over all predicted BBs.
    ref_fps = [_smiles_to_fp(s) for s in ref_leaves]
    ref_fps = [fp for fp in ref_fps if fp is not None]
    pred_fps = [(s, _smiles_to_fp(s)) for s in pred_leaves]

    tanimoto_scores = []
    for smi, pfp in pred_fps:
        if pfp is None or not ref_fps:
            tanimoto_scores.append(0.0)
            continue
        max_sim = max(DataStructs.TanimotoSimilarity(pfp, rfp) for rfp in ref_fps)
        tanimoto_scores.append(max_sim)
    avg_max_tanimoto = sum(tanimoto_scores) / len(tanimoto_scores) if tanimoto_scores else 0.0

    return {
        "jaccard": round(jaccard, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "max_tanimoto": round(avg_max_tanimoto, 4),
        "n_ref": len(ref_leaves),
        "n_pred": len(pred_leaves),
        "n_common": len(common),
    }


def get_retro_node_reactions(node):
    """Extract the list of reactions (disconnections) from a RetroNode DAG.

    Returns:
        list of dicts with reaction name, bond_type, and child SMILES.
    """
    reactions = []
    if node is None:
        return reactions
    if not getattr(node, "is_leaf", True) and node.children:
        rxn_info = getattr(node, "reaction_info", None)
        bond = getattr(node, "bond_type", None)
        reactions.append({
            "parent": node.smiles[:60],
            "reaction": rxn_info.name if rxn_info else "BRICS",
            "bond_type": f"{bond[0]}-{bond[1]}" if bond else None,
            "children": [c.smiles[:60] for c in node.children],
        })
        for c in node.children:
            reactions.extend(get_retro_node_reactions(c))
    return reactions


def get_retro_node_leaf_sets(node):
    if node is None:
        return []
    if getattr(node, "is_leaf", False):
        bbs = getattr(node, "bb_smiles", None)
        if bbs:
            can_bbs = set()
            for bb in bbs:
                mol = Chem.MolFromSmiles(bb)
                if mol:
                    can_bbs.add(Chem.MolToSmiles(mol))
                else:
                    can_bbs.add(bb)
            return [can_bbs]
        # Fallback to fragment smiles
        mol = Chem.MolFromSmiles(node.smiles)
        if mol:
            return [{Chem.MolToSmiles(mol)}]
        return [{node.smiles}]
    leaf_sets = []
    for c in getattr(node, "children", []):
        leaf_sets.extend(get_retro_node_leaf_sets(c))
    return leaf_sets


def _init_worker(mol_props_path, fp_size, tiers, timeout, max_depth, max_nodes, reference_routes=None):
    """Initialize per-worker globals (called once per pool process)."""
    global _worker_cf, _worker_lib, _worker_tiers
    global _worker_timeout, _worker_max_depth, _worker_max_nodes, _worker_reference_routes
    _worker_cf = load_compound_filter(mol_props_path, fpSize=fp_size)
    _worker_lib = ReactionLibrary.default() if any(t in tiers for t in ["2", "3"]) else None
    _worker_tiers = tiers
    _worker_timeout = timeout
    _worker_max_depth = max_depth
    _worker_max_nodes = max_nodes
    _worker_reference_routes = reference_routes


# ---------------------------------------------------------------------------
# Per-tier runners (only T3 remains standalone — it uses a different engine)
# ---------------------------------------------------------------------------


def run_tier3(smiles, cf, lib, timeout, max_depth=5, max_nodes=2000):
    t0 = time.time()
    try:
        from fragmentretro.smarts_retro import SmartsRetrosynthesis

        retro = SmartsRetrosynthesis(
            lib, 
            max_depth=max_depth, 
            max_nodes=max_nodes,
            min_bb_heavy_atoms=getattr(cf, "min_heavy_atoms", 0),
            max_bb_heavy_atoms=getattr(cf, "max_heavy_atoms", 100)
        )

        def is_purchasable(smi):
            try:
                return cf.has_match(smi)
            except Exception:
                return False

        routes = retro.retrosynthesise(smiles, is_purchasable=is_purchasable,
                                        max_routes=5)

        routes_info = []
        for r in routes:
            routes_info.append({
                "solved": r.is_solved,
                "steps": r.num_steps,
                "lls": r.longest_linear_sequence,
                "n_bbs": r.num_leaves,
                "avg_reliability": round(r.avg_reliability, 4),
                "reaction": r.reaction.name if r.reaction else None,
            })

        best_route_solved = (routes_info[0]["solved"] if routes_info else False)

        # Derive score from best route
        best_route_leaves = []
        best_route_reactions = []
        if routes:
            best = max(routes, key=lambda r: (r.is_solved, r.avg_reliability))
            for n in _iter_leaves_t3(best):
                mol = Chem.MolFromSmiles(n.smiles)
                if mol:
                    best_route_leaves.append(Chem.MolToSmiles(mol))
                else:
                    best_route_leaves.append(n.smiles)
            # Extract reactions from the best route tree
            best_route_reactions = _collect_t3_reactions(best)
            if best.is_solved:
                reliability_score = best.avg_reliability
                steps = best.num_steps
                efficiency_score = 1.0 / (1.0 + 0.2 * steps)
                score = 0.5 * reliability_score + 0.5 * efficiency_score
            else:
                total_leaves = best.num_leaves
                bb_leaves = sum(1 for n in _iter_leaves_t3(best) if n.is_building_block)
                partial = bb_leaves / total_leaves if total_leaves > 0 else 0.0
                score = 0.3 * partial * best.avg_reliability
        else:
            score = 0.0

        return {
            "tier": "3_smarts", "solved": best_route_solved,
            "score": round(score, 6),
            "n_routes": len(routes_info),
            "best_route": routes_info[0] if routes_info else None,
            "predicted_leaves": list(set(best_route_leaves)),
            "reactions": best_route_reactions,
            "time_s": round(time.time() - t0, 4),
        }
    except Exception as e:
        return {"tier": "3_smarts", "solved": False, "score": 0.0,
                "n_routes": 0, "best_route": None, "predicted_leaves": [],
                "reactions": [],
                "time_s": round(time.time() - t0, 4), "error": str(e)}


def _iter_leaves_t3(node):
    """Iterate leaf nodes of a RetroSynthNode tree."""
    if node.is_leaf:
        yield node
    else:
        for c in node.children:
            yield from _iter_leaves_t3(c)


def _collect_t3_reactions(node):
    """Extract reactions from a RetroSynthNode tree."""
    reactions = []
    if node.is_leaf:
        return reactions
    rxn = getattr(node, "reaction", None)
    reactions.append({
        "parent": node.smiles[:60],
        "reaction": rxn.name if rxn else None,
        "class": rxn.reaction_class if rxn else None,
        "reliability": round(rxn.reliability, 4) if rxn else None,
        "children": [c.smiles[:60] for c in node.children],
    })
    for c in node.children:
        reactions.extend(_collect_t3_reactions(c))
    return reactions


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _process_molecule(task):
    """Process a single molecule (used by both serial and parallel modes).

    Runs ``_run_retro`` **once** for all BRICS-based tiers (T1b/T1c/T1d/T2)
    via ``compute_all_brics_tiers``, then runs T3 independently (different engine).

    Args:
        task: tuple of (idx, smiles, n_total) for progress display.

    Returns:
        dict with per-molecule results.
    """
    idx, smiles, n_total = task
    cf = _worker_cf
    lib = _worker_lib
    tiers = _worker_tiers
    timeout = _worker_timeout
    max_depth = _worker_max_depth
    max_nodes = _worker_max_nodes
    ref_routes = _worker_reference_routes

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"idx": idx, "smiles": smiles, "valid": False, "tiers": {}}

    canonical = Chem.MolToSmiles(mol)
    mol_result = {
        "idx": idx,
        "smiles": canonical,
        "original_smiles": smiles,
        "valid": True,
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "tiers": {},
    }

    # --- BRICS tiers: single _run_retro call via compute_all_brics_tiers ---
    need_binary = "1" in tiers or "1b" in tiers
    need_continuous = "1" in tiers or "1c" in tiers
    need_dag = "1" in tiers or "1d" in tiers
    need_validated = "2" in tiers

    if need_binary or need_continuous or need_dag or need_validated:
        t0 = time.time()
        try:
            brics_result = compute_all_brics_tiers(
                canonical,
                cf,
                reaction_library=lib,
                include_binary=need_binary,
                include_continuous=need_continuous,
                include_dag=need_dag,
                include_validated=need_validated,
            )
            brics_time = time.time() - t0

            # Distribute the shared compute time proportionally across tiers
            n_brics_tiers = sum([need_binary, need_continuous, need_dag, need_validated])
            per_tier_time = round(brics_time / n_brics_tiers, 4) if n_brics_tiers else 0

            if need_binary:
                mol_result["tiers"]["1_binary"] = {
                    "tier": "1_binary",
                    "solved": brics_result.binary,
                    "score": 1.0 if brics_result.binary else 0.0,
                    "time_s": per_tier_time,
                }

            if need_continuous:
                mol_result["tiers"]["1_continuous"] = {
                    "tier": "1_continuous",
                    "solved": brics_result.continuous_score > 0,
                    "score": round(brics_result.continuous_score, 6),
                    "time_s": per_tier_time,
                }

            if need_dag:
                dr = brics_result.dag_result
                pred_leaf_sets = get_retro_node_leaf_sets(dr.dag) if dr.feasible and dr.dag else []
                pred_leaves_rep = [list(s)[0] for s in pred_leaf_sets if s]
                mol_result["tiers"]["1_dag"] = {
                    "tier": "1_dag",
                    "solved": dr.feasible,
                    "score": round(dr.score, 6),
                    "total_steps": dr.total_steps,
                    "lls": dr.longest_linear_sequence,
                    "num_bbs": dr.num_building_blocks,
                    "convergence": round(dr.convergence_score, 4),
                    "step_score": round(dr.step_score, 4),
                    "availability_score": round(dr.availability_score, 4),
                    "bond_feasibility_score": round(dr.bond_feasibility_score, 4),
                    "predicted_leaves": pred_leaves_rep,
                    "reactions": get_retro_node_reactions(dr.dag) if dr.feasible and dr.dag else [],
                    "time_s": per_tier_time,
                }
                
                ref_leaves = ref_routes.get(canonical) if ref_routes else None
                if ref_leaves is not None:
                    mol_result["tiers"]["1_dag"]["route_similarity"] = compute_route_similarity(set(ref_leaves), set(pred_leaves_rep))

            if need_validated and brics_result.validated_result is not None:
                vr = brics_result.validated_result
                pred_leaf_sets = get_retro_node_leaf_sets(vr.dag) if vr.feasible and vr.dag else []
                pred_leaves_rep = [list(s)[0] for s in pred_leaf_sets if s]
                mol_result["tiers"]["2_validated"] = {
                    "tier": "2_validated",
                    "solved": vr.feasible and vr.validation_coverage >= 1.0,
                    "score": round(vr.score, 6),
                    "total_steps": vr.total_steps,
                    "lls": vr.longest_linear_sequence,
                    "num_bbs": vr.num_building_blocks,
                    "convergence": round(vr.convergence_score, 4),
                    "validation_coverage": round(vr.validation_coverage, 4),
                    "matched_reactions": [
                        {"name": name, "score": round(s, 4)}
                        for name, s in vr.matched_reactions
                    ],
                    "n_matched": len(vr.matched_reactions),
                    "constraints_satisfied": vr.constraints_satisfied,
                    "predicted_leaves": pred_leaves_rep,
                    "reactions": get_retro_node_reactions(vr.dag) if vr.feasible and vr.dag else [],
                    "time_s": per_tier_time,
                }
                
                ref_leaves = ref_routes.get(canonical) if ref_routes else None
                if ref_leaves is not None:
                    mol_result["tiers"]["2_validated"]["route_similarity"] = compute_route_similarity(set(ref_leaves), set(pred_leaves_rep))

        except Exception as e:
            # Fallback: mark all requested BRICS tiers as failed
            for tier_key, needed in [("1_binary", need_binary), ("1_continuous", need_continuous),
                                     ("1_dag", need_dag), ("2_validated", need_validated)]:
                if needed:
                    mol_result["tiers"][tier_key] = {
                        "tier": tier_key, "solved": False, "score": 0.0,
                        "time_s": round(time.time() - t0, 4), "error": str(e),
                    }

    # --- T3: separate engine (SMARTS retrosynthesis) ---
    if "3" in tiers:
        t3_res = run_tier3(canonical, cf, lib, timeout, max_depth, max_nodes)
        ref_leaves = ref_routes.get(canonical) if ref_routes else None
        if ref_leaves is not None and "predicted_leaves" in t3_res:
             pred_set = set(t3_res["predicted_leaves"])
             t3_res["route_similarity"] = compute_route_similarity(set(ref_leaves), pred_set)
        mol_result["tiers"]["3_smarts"] = t3_res

    return mol_result


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(targets, config, tiers, timeout=120.0,
                  max_depth=5, max_nodes=2000, n_workers=1, reference_routes=None):
    mol_props_path = config["fragmentretro_stock"]
    fp_size = config.get("fp_size", 2048)

    if n_workers > 1:
        return _run_benchmark_parallel(targets, config, tiers, timeout,
                                        max_depth, max_nodes, n_workers, reference_routes)

    # --- Serial mode ---
    print(f"Loading CompoundFilter from {mol_props_path} ...")
    cf = load_compound_filter(mol_props_path, fpSize=fp_size)
    print(f"  {cf.len_BBs} building blocks loaded")

    lib = None
    if any(t in tiers for t in ["2", "3"]):
        print("Loading ReactionLibrary (91 reactions) ...")
        lib = ReactionLibrary.default()
        print(f"  {len(lib)} reactions loaded")

    # Set globals for _process_molecule
    global _worker_cf, _worker_lib, _worker_tiers
    global _worker_timeout, _worker_max_depth, _worker_max_nodes, _worker_reference_routes
    _worker_cf = cf
    _worker_lib = lib
    _worker_tiers = tiers
    _worker_timeout = timeout
    _worker_max_depth = max_depth
    _worker_max_nodes = max_nodes
    _worker_reference_routes = reference_routes

    results = []
    n_total = len(targets)

    for idx, smiles in enumerate(targets):
        mol_result = _process_molecule((idx, smiles, n_total))

        if not mol_result["valid"]:
            print(f"  [{idx+1}/{n_total}] SKIP (invalid): {smiles[:60]}")
        else:
            tier_labels = []
            for tier_key in sorted(mol_result["tiers"]):
                t = mol_result["tiers"][tier_key]
                if tier_key == "1_binary":
                    tier_labels.append(f"T1b={'Y' if t['solved'] else 'N'}")
                else:
                    tier_labels.append(f"{_tier_label(tier_key)}={t['score']:.3f}")
            total_time = sum(t.get("time_s", 0) for t in mol_result["tiers"].values())
            status = " | ".join(tier_labels)
            canonical = mol_result["smiles"]
            print(f"  [{idx+1}/{n_total}] {canonical[:50]:50s}  {status}  ({total_time:.2f}s)")

        results.append(mol_result)

    return results


def _run_benchmark_parallel(targets, config, tiers, timeout,
                            max_depth, max_nodes, n_workers, reference_routes=None):
    """Run benchmark using multiprocessing pool."""
    import multiprocessing as mp

    mol_props_path = config["fragmentretro_stock"]
    fp_size = config.get("fp_size", 2048)

    print(f"Parallel mode: {n_workers} workers")
    print(f"Loading CompoundFilter + ReactionLibrary in each worker ...")

    n_total = len(targets)
    tasks = [(idx, smi, n_total) for idx, smi in enumerate(targets)]

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(mol_props_path, fp_size, tiers, timeout, max_depth, max_nodes, reference_routes),
    ) as pool:
        results_unordered = []
        for mol_result in pool.imap_unordered(_process_molecule, tasks, chunksize=4):
            idx = mol_result["idx"]
            results_unordered.append(mol_result)
            done = len(results_unordered)

            if not mol_result["valid"]:
                print(f"  [{done}/{n_total}] SKIP (invalid): {mol_result['smiles'][:60]}")
            else:
                tier_labels = []
                for tier_key in sorted(mol_result["tiers"]):
                    t = mol_result["tiers"][tier_key]
                    if tier_key == "1_binary":
                        tier_labels.append(f"T1b={'Y' if t['solved'] else 'N'}")
                    else:
                        tier_labels.append(f"{_tier_label(tier_key)}={t['score']:.3f}")
                total_time = sum(t.get("time_s", 0) for t in mol_result["tiers"].values())
                status = " | ".join(tier_labels)
                canonical = mol_result["smiles"]
                print(f"  [{done}/{n_total}] {canonical[:50]:50s}  {status}  ({total_time:.2f}s)")

    # Re-sort by original index
    results_unordered.sort(key=lambda r: r["idx"])
    return results_unordered


def _tier_label(tier_key):
    """Short label for progress display."""
    return {
        "1_binary": "T1b", "1_continuous": "T1c", "1_dag": "T1d",
        "2_validated": "T2", "3_smarts": "T3",
    }.get(tier_key, tier_key)


def compute_summary(results):
    valid = [r for r in results if r["valid"]]
    n = len(valid)
    summary = {"n_targets": len(results), "n_valid": n}

    tier_names = set()
    for r in valid:
        tier_names.update(r["tiers"].keys())

    for tier in sorted(tier_names):
        tier_results = [r["tiers"][tier] for r in valid if tier in r["tiers"]]
        if not tier_results:
            continue

        solved = sum(1 for t in tier_results if t.get("solved", False))
        scores = [t["score"] for t in tier_results]
        times = [t["time_s"] for t in tier_results]

        stats = {
            "n_molecules": len(tier_results),
            "solved": solved,
            "solve_rate": round(solved / len(tier_results), 4) if tier_results else 0,
            "mean_score": round(sum(scores) / len(scores), 4),
            "median_score": round(sorted(scores)[len(scores)//2], 4),
            "mean_time_s": round(sum(times) / len(times), 4),
            "median_time_s": round(sorted(times)[len(times)//2], 4),
            "total_time_s": round(sum(times), 2),
            "min_time_s": round(min(times), 4),
            "max_time_s": round(max(times), 4),
        }

        steps = [t["total_steps"] for t in tier_results if "total_steps" in t and t.get("solved")]
        lls_vals = [t["lls"] for t in tier_results if "lls" in t and t.get("solved")]
        bbs = [t["num_bbs"] for t in tier_results if "num_bbs" in t and t.get("solved")]

        if steps:
            stats["mean_steps"] = round(sum(steps) / len(steps), 2)
        if lls_vals:
            stats["mean_lls"] = round(sum(lls_vals) / len(lls_vals), 2)
        if bbs:
            stats["mean_bbs"] = round(sum(bbs) / len(bbs), 2)

        coverages = [t["validation_coverage"] for t in tier_results
                     if "validation_coverage" in t and t.get("solved")]
        if coverages:
            stats["mean_validation_coverage"] = round(sum(coverages) / len(coverages), 4)

        # Route similarity aggregation
        sim_results = [t["route_similarity"] for t in tier_results if "route_similarity" in t]
        if sim_results:
            for metric in ["jaccard", "recall", "precision", "max_tanimoto"]:
                vals = [s[metric] for s in sim_results]
                stats[f"mean_{metric}"] = round(sum(vals) / len(vals), 4)
            stats["exact_match_rate"] = round(
                sum(1 for s in sim_results if s["jaccard"] >= 1.0) / len(sim_results), 4
            )

        errors = sum(1 for t in tier_results if "error" in t)
        if errors:
            stats["errors"] = errors

        summary[tier] = stats

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark FragmentRetro")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to benchmark_config.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--tiers", type=str, nargs="+", default=["1", "2", "3"],
                        help="Tiers to run: 1 1b 1c 1d 2 3 (default: 1 2 3)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Timeout per molecule (seconds)")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Limit to first N targets")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Tier 3 max tree depth")
    parser.add_argument("--max-nodes", type=int, default=2000,
                        help="Tier 3 max nodes to explore")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to reference routes JSON (e.g. n1-routes.json) for exact match comparison")
    parser.add_argument("--workers", "-j", type=int, default=1,
                        help="Number of parallel workers (default: 1 = serial). "
                             "Each worker loads its own CompoundFilter and "
                             "ReactionLibrary. Use -j 0 for auto (all CPUs).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    targets = []
    with open(config["targets_file"]) as f:
        for line in f:
            smi = line.strip()
            if smi:
                targets.append(smi)

    if args.max_targets:
        targets = targets[:args.max_targets]

    n_workers = args.workers
    if n_workers == 0:
        import os
        n_workers = os.cpu_count() or 1
    if n_workers < 0:
        n_workers = 1

    print(f"FragmentRetro Benchmark")
    print(f"  Targets:    {len(targets)}")
    print(f"  Stock BBs:  {config['n_stock_bbs']}")
    print(f"  Tiers:      {args.tiers}")
    print(f"  Timeout:    {args.timeout}s/mol")
    print(f"  Workers:    {n_workers}" + (" (parallel)" if n_workers > 1 else " (serial)"))
    print()

    reference_routes = None
    if args.reference:
        reference_routes = load_reference_routes(args.reference)

    t_start = time.time()
    results = run_benchmark(targets, config, args.tiers,
                            timeout=args.timeout,
                            max_depth=args.max_depth,
                            max_nodes=args.max_nodes,
                            n_workers=n_workers,
                            reference_routes=reference_routes)
    total_time = time.time() - t_start

    summary = compute_summary(results)
    summary["total_benchmark_time_s"] = round(total_time, 2)
    summary["tool"] = "FragmentRetro"
    summary["tiers_run"] = args.tiers
    summary["config"] = config

    output_path = args.output or str(
        Path(config["targets_file"]).parent / "results_fragmentretro.json"
    )
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FRAGMENTRETRO BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Targets: {summary['n_valid']}/{summary['n_targets']} valid")
    print(f"Total time: {total_time:.1f}s")
    print()

    for tier in sorted(k for k in summary if k.startswith(("1_", "2_", "3_"))):
        s = summary[tier]
        s_str = (f"  {tier:20s}  solve={s['solve_rate']:.1%}  "
              f"score={s['mean_score']:.3f}  "
              f"time={s['mean_time_s']:.3f}s/mol  "
              f"(total={s['total_time_s']:.1f}s)")
        if "mean_jaccard" in s:
            s_str += f" J={s['mean_jaccard']:.3f} R={s['mean_recall']:.3f} T={s['mean_max_tanimoto']:.3f}"
        print(s_str)
        if "mean_steps" in s:
            print(f"  {'':20s}  steps={s['mean_steps']:.1f}  "
                  f"lls={s.get('mean_lls', 'N/A')}  "
                  f"bbs={s.get('mean_bbs', 'N/A')}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
