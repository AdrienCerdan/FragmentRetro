"""Feasibility scoring for FragmentRetro.

Provides binary (solved/unsolved) and continuous scoring functions
suitable for standalone use and as reward signals in Reinvent4 RL.
"""

import math
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
