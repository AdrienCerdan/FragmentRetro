"""Feasibility scoring for FragmentRetro.

Provides binary (solved/unsolved) and continuous scoring functions
suitable for standalone use and as reward signals in Reinvent4 RL.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from fragmentretro.bond_feasibility import get_solution_bond_feasibility
from fragmentretro.fragmenter import BRICSFragmenter, rBRICSFragmenter
from fragmentretro.fragmenter_base import Fragmenter
from fragmentretro.retrosynthesis import Retrosynthesis
from fragmentretro.solutions import RetrosynthesisSolution
from fragmentretro.typing import SolutionType
from fragmentretro.utils.filter_compound import CompoundFilter
from fragmentretro.utils.logging_config import logger


def _run_retro(
    smiles: str,
    compound_filter: CompoundFilter,
    binary_mode: bool = False,
    use_rbrics: bool = True,
    solution_cap: int | None = None,
) -> tuple[Retrosynthesis, RetrosynthesisSolution]:
    """Internal helper to run FragmentRetro on a single SMILES.

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter (shared singleton).
        binary_mode: If True, use early termination in substructure matching.
        use_rbrics: If True, use r-BRICS; otherwise use BRICS.
        solution_cap: Maximum number of solutions to enumerate.

    Returns:
        Tuple of (Retrosynthesis, RetrosynthesisSolution).
    """
    fragmenter: Fragmenter
    if use_rbrics:
        fragmenter = rBRICSFragmenter(smiles)
    else:
        fragmenter = BRICSFragmenter(smiles)

    retro_tool = Retrosynthesis(
        fragmenter,
        compound_filter=compound_filter,
        mol_properties_path=compound_filter.mol_properties_path,
        binary_mode=binary_mode,
    )
    retro_tool.fragment_retrosynthesis()

    retro_solution = RetrosynthesisSolution(retro_tool)
    retro_solution.fill_solutions(solution_cap=solution_cap)

    return retro_tool, retro_solution


def is_feasible(
    smiles: str,
    compound_filter: CompoundFilter,
    use_rbrics: bool = True,
) -> bool:
    """Binary feasibility check. Returns True if at least one solution exists.

    Uses binary_mode for maximum speed (early termination on first BB match).

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter.
        use_rbrics: Use r-BRICS (recommended) or BRICS.

    Returns:
        True if the molecule can be reconstructed from building blocks.
    """
    try:
        _, retro_solution = _run_retro(
            smiles, compound_filter, binary_mode=True, use_rbrics=use_rbrics, solution_cap=1
        )
        return len(retro_solution.solutions) > 0
    except Exception as e:
        logger.warning(f"[Scoring] Failed to score {smiles}: {e}")
        return False


def compute_score(
    smiles: str,
    compound_filter: CompoundFilter,
    use_rbrics: bool = True,
    solution_cap: int = 5,
    step_weight: float = 0.35,
    availability_weight: float = 0.30,
    feasibility_weight: float = 0.35,
) -> float:
    """Continuous feasibility score in [0, 1]. Higher = more feasible.

    Components:
        - step_score: fewer fragments in best solution = fewer synthetic steps = better
        - availability_score: more BB matches per fragment = more robust supply chain
        - bond_feasibility_score: BRICS bond types mapped to reaction reliability

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        solution_cap: Max solutions to enumerate (lower = faster).
        step_weight: Weight for step score component.
        availability_weight: Weight for BB availability component.
        feasibility_weight: Weight for bond feasibility component.

    Returns:
        Score in [0.0, 1.0].
    """
    try:
        retro_tool, retro_solution = _run_retro(
            smiles, compound_filter, binary_mode=False, use_rbrics=use_rbrics, solution_cap=solution_cap
        )
    except Exception as e:
        logger.warning(f"[Scoring] Failed to score {smiles}: {e}")
        return 0.0

    if not retro_solution.solutions:
        return 0.0

    # Pick best solution (fewest fragments)
    best_solution: SolutionType = min(retro_solution.solutions, key=len)
    num_frags = len(best_solution)
    max_frags = retro_tool.num_fragments

    # 1) Step score: fewer fragments = better
    if max_frags <= 1:
        step_score = 1.0
    else:
        step_score = 1.0 - (num_frags - 1) / (max_frags - 1)

    # 2) BB availability: log-scaled min BB count across fragments
    bb_counts = []
    for comb in best_solution:
        bbs = retro_tool.comb_bbs_dict.get(comb, set())
        bb_counts.append(len(bbs))
    min_bb_count = min(bb_counts) if bb_counts else 0
    availability_score = min(1.0, math.log1p(min_bb_count) / math.log1p(100))

    # 3) Bond feasibility: average reconnection reliability
    bond_types = retro_tool.fragmenter.get_bond_types_for_solution(best_solution)
    bond_feasibility_score = get_solution_bond_feasibility(bond_types)

    total = (
        step_weight * step_score
        + availability_weight * availability_score
        + feasibility_weight * bond_feasibility_score
    )

    logger.debug(
        f"[Scoring] {smiles}: steps={step_score:.2f}, avail={availability_score:.2f}, "
        f"bond_feas={bond_feasibility_score:.2f}, total={total:.2f}"
    )

    return total


def load_compound_filter(mol_properties_path: str | Path, fpSize: int = 2048) -> CompoundFilter:
    """Convenience function to load a CompoundFilter singleton.

    Call this once at startup and pass the result to scoring functions.

    Args:
        mol_properties_path: Path to precomputed properties JSON.
        fpSize: Fingerprint size (must match precomputation).

    Returns:
        CompoundFilter instance.
    """
    return CompoundFilter(Path(mol_properties_path), fpSize=fpSize)


@dataclass
class DAGScoreResult:
    """Rich result from DAG-based scoring.

    Attributes:
        score: Overall score in [0, 1].
        feasible: Whether any solution was found.
        total_steps: Number of synthetic steps in best route.
        longest_linear_sequence: Critical path length (LLS).
        num_building_blocks: Number of BBs (leaves).
        convergence_score: 1.0=fully convergent, 0.0=fully linear.
        step_score: Component score for step count.
        availability_score: Component score for BB availability.
        bond_feasibility_score: Component score for reaction reliability.
        dag: The RetroNode tree (None if not feasible).
    """

    score: float
    feasible: bool
    total_steps: int = 0
    longest_linear_sequence: int = 0
    num_building_blocks: int = 0
    convergence_score: float = 0.0
    step_score: float = 0.0
    availability_score: float = 0.0
    bond_feasibility_score: float = 0.0
    dag: object = None  # RetroNode, typed as object to keep import lazy


def compute_score_with_dag(
    smiles: str,
    compound_filter: CompoundFilter,
    use_rbrics: bool = True,
    solution_cap: int = 5,
    step_weight: float = 0.25,
    availability_weight: float = 0.20,
    feasibility_weight: float = 0.25,
    convergence_weight: float = 0.15,
    lls_weight: float = 0.15,
) -> DAGScoreResult:
    """Continuous scoring with DAG analysis. Returns rich result with route details.

    Extends compute_score with two additional components derived from the DAG:
        - convergence_score: convergent routes score higher (parallel synthesis)
        - lls_score: shorter longest-linear-sequence scores higher

    This is more expensive than compute_score because it builds the DAG.
    Use compute_score when you only need a number, use this when you need
    the route tree or convergence information.

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        solution_cap: Max solutions to enumerate.
        step_weight: Weight for fragment count score.
        availability_weight: Weight for BB availability score.
        feasibility_weight: Weight for bond feasibility score.
        convergence_weight: Weight for convergence score.
        lls_weight: Weight for LLS score.

    Returns:
        DAGScoreResult with score, route metrics, and the DAG tree.
    """
    from fragmentretro.retro_dag import build_best_dag

    try:
        retro_tool, retro_solution = _run_retro(
            smiles, compound_filter, binary_mode=False, use_rbrics=use_rbrics, solution_cap=solution_cap
        )
    except Exception as e:
        logger.warning(f"[Scoring] Failed to score {smiles}: {e}")
        return DAGScoreResult(score=0.0, feasible=False)

    if not retro_solution.solutions:
        return DAGScoreResult(score=0.0, feasible=False)

    # Pick best solution (fewest fragments)
    best_solution: SolutionType = min(retro_solution.solutions, key=len)
    num_frags = len(best_solution)
    max_frags = retro_tool.num_fragments

    # 1) Step score (same as compute_score)
    if max_frags <= 1:
        step_score = 1.0
    else:
        step_score = 1.0 - (num_frags - 1) / (max_frags - 1)

    # 2) BB availability (same as compute_score)
    bb_counts = []
    for comb in best_solution:
        bbs = retro_tool.comb_bbs_dict.get(comb, set())
        bb_counts.append(len(bbs))
    min_bb_count = min(bb_counts) if bb_counts else 0
    availability_score = min(1.0, math.log1p(min_bb_count) / math.log1p(100))

    # 3) Bond feasibility (same as compute_score)
    bond_types = retro_tool.fragmenter.get_bond_types_for_solution(best_solution)
    bond_feasibility_score = get_solution_bond_feasibility(bond_types)

    # 4+5) DAG-derived metrics
    try:
        dag = build_best_dag(best_solution, retro_tool.fragmenter, retro_tool.comb_bbs_dict)
        convergence = dag.convergence_score
        lls = dag.longest_linear_sequence
        total_steps = dag.num_steps
        n_bbs = dag.num_leaves

        # LLS score: lower LLS relative to num_frags = better
        # Best possible LLS for n_bbs is ceil(log2(n_bbs))
        if n_bbs <= 1:
            lls_score = 1.0
        else:
            worst_lls = n_bbs - 1
            lls_score = 1.0 - (lls - 1) / max(worst_lls - 1, 1)
            lls_score = max(0.0, min(1.0, lls_score))
    except Exception as e:
        logger.warning(f"[Scoring] DAG construction failed for {smiles}: {e}")
        convergence = 0.5
        lls_score = 0.5
        lls = num_frags - 1
        total_steps = num_frags - 1
        n_bbs = num_frags
        dag = None

    total = (
        step_weight * step_score
        + availability_weight * availability_score
        + feasibility_weight * bond_feasibility_score
        + convergence_weight * convergence
        + lls_weight * lls_score
    )

    logger.debug(
        f"[Scoring+DAG] {smiles}: steps={step_score:.2f}, avail={availability_score:.2f}, "
        f"bond_feas={bond_feasibility_score:.2f}, conv={convergence:.2f}, "
        f"lls_score={lls_score:.2f}, total={total:.2f}"
    )

    return DAGScoreResult(
        score=total,
        feasible=True,
        total_steps=total_steps,
        longest_linear_sequence=lls,
        num_building_blocks=n_bbs,
        convergence_score=convergence,
        step_score=step_score,
        availability_score=availability_score,
        bond_feasibility_score=bond_feasibility_score,
        dag=dag,
    )
