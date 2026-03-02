"""Feasibility scoring for FragmentRetro.

Provides binary (solved/unsolved) and continuous scoring functions
suitable for standalone use and as reward signals in Reinvent4 RL.

Tiered scoring:
  - Tier 1: BRICS-only (fast, existing behavior)
  - Tier 2: BRICS + SMARTS validation (medium, more accurate)
  - Tier 3: SMARTS-only retrosynthesis (slower, broadest coverage)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fragmentretro.bond_feasibility import get_solution_bond_feasibility
from fragmentretro.constraints import ConstraintConfig, build_constraints
from fragmentretro.fragmenter import BRICSFragmenter, rBRICSFragmenter
from fragmentretro.fragmenter_base import Fragmenter
from fragmentretro.reaction_library import ReactionLibrary
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
        matched_reactions: List of (reaction_name, reliability) for SMARTS-validated bonds.
        validation_coverage: Fraction of bonds validated by SMARTS (0-1).
        constraints_satisfied: Whether all constraints are met.
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
    matched_reactions: list[tuple[str, float]] = field(default_factory=list)
    validation_coverage: float = 0.0
    constraints_satisfied: bool = True


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


# ---------------------------------------------------------------------------
# Tier 2: BRICS + SMARTS validation
# ---------------------------------------------------------------------------


def _validate_dag_with_smarts(
    dag: object,  # RetroNode
    library: ReactionLibrary,
) -> tuple[list[tuple[str, float]], float]:
    """Validate each disconnection in a DAG against the reaction library.

    Returns (matched_reactions, coverage) where coverage is the fraction
    of bonds that matched a known reaction.
    """
    from fragmentretro.retro_dag import RetroNode

    node: RetroNode = dag  # type: ignore[assignment]
    matched: list[tuple[str, float]] = []
    total_bonds = 0

    def _walk(n: RetroNode) -> None:
        nonlocal total_bonds
        if n.is_leaf or not n.children:
            return
        total_bonds += 1

        child_smiles = [c.smiles for c in n.children]
        is_valid, rxn, score = library.validate_bond_disconnection(
            n.smiles, child_smiles, n.bond_type
        )
        if is_valid and rxn is not None:
            matched.append((rxn.name, score))

        for c in n.children:
            _walk(c)

    _walk(node)

    coverage = len(matched) / total_bonds if total_bonds > 0 else 1.0
    return matched, coverage


def compute_score_validated(
    smiles: str,
    compound_filter: CompoundFilter,
    reaction_library: ReactionLibrary,
    *,
    # Simple constraint params
    allowed_reactions: list[str] | None = None,
    blocked_reactions: list[str] | None = None,
    allowed_classes: list[str] | None = None,
    max_steps: int | None = None,
    max_lls: int | None = None,
    min_reliability: float = 0.0,
    require_all_validated: bool = False,
    # Or a full config object
    constraints: ConstraintConfig | None = None,
    # Scoring weights
    use_rbrics: bool = True,
    solution_cap: int = 5,
    step_weight: float = 0.20,
    availability_weight: float = 0.15,
    feasibility_weight: float = 0.25,
    convergence_weight: float = 0.10,
    lls_weight: float = 0.10,
    validation_weight: float = 0.20,
) -> DAGScoreResult:
    """Tier 2: BRICS decomposition + SMARTS validation scoring.

    Like compute_score_with_dag but adds a validation_score component
    that rewards disconnections matching known reactions from the library.

    The validation_weight component measures what fraction of bonds in the
    best DAG match a real reaction in the library, weighted by reliability.

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter.
        reaction_library: Loaded ReactionLibrary for SMARTS validation.
        allowed_reactions: Whitelist reaction names (simple API).
        blocked_reactions: Blacklist reaction names (simple API).
        allowed_classes: Whitelist reaction classes (simple API).
        max_steps: Maximum synthesis steps (simple API).
        max_lls: Maximum LLS (simple API).
        min_reliability: Minimum reaction reliability (simple API).
        require_all_validated: Require all bonds match library (simple API).
        constraints: Full ConstraintConfig (overrides simple params).
        use_rbrics: Use r-BRICS fragmentation.
        solution_cap: Max solutions to enumerate.
        step_weight .. validation_weight: Scoring component weights.

    Returns:
        DAGScoreResult with SMARTS validation data.
    """
    from fragmentretro.retro_dag import build_best_dag

    # Build constraints
    cfg = build_constraints(
        constraints=constraints,
        allowed_reactions=allowed_reactions,
        blocked_reactions=blocked_reactions,
        allowed_classes=allowed_classes,
        max_steps=max_steps,
        max_lls=max_lls,
        min_reliability=min_reliability,
        require_all_validated=require_all_validated,
    )

    # Filter library if constraints provided
    active_lib = reaction_library
    if cfg:
        active_lib = cfg.get_filtered_library(reaction_library)

    # Run BRICS retrosynthesis (Tier 1 core)
    try:
        retro_tool, retro_solution = _run_retro(
            smiles, compound_filter, binary_mode=False,
            use_rbrics=use_rbrics, solution_cap=solution_cap,
        )
    except Exception as e:
        logger.warning(f"[Scoring] Failed to score {smiles}: {e}")
        return DAGScoreResult(score=0.0, feasible=False)

    if not retro_solution.solutions:
        return DAGScoreResult(score=0.0, feasible=False)

    best_solution: SolutionType = min(retro_solution.solutions, key=len)
    num_frags = len(best_solution)
    max_frags = retro_tool.num_fragments

    # Step score
    step_score = 1.0 if max_frags <= 1 else 1.0 - (num_frags - 1) / (max_frags - 1)

    # BB availability
    bb_counts = []
    for comb in best_solution:
        bbs = retro_tool.comb_bbs_dict.get(comb, set())
        bb_counts.append(len(bbs))
    min_bb_count = min(bb_counts) if bb_counts else 0
    availability_score = min(1.0, math.log1p(min_bb_count) / math.log1p(100))

    # Bond feasibility (heuristic baseline)
    bond_types = retro_tool.fragmenter.get_bond_types_for_solution(best_solution)
    bond_feasibility_score = get_solution_bond_feasibility(bond_types)

    # Build DAG
    matched_reactions: list[tuple[str, float]] = []
    validation_coverage = 0.0
    constraints_satisfied = True

    try:
        dag = build_best_dag(best_solution, retro_tool.fragmenter, retro_tool.comb_bbs_dict)
        convergence = dag.convergence_score
        lls = dag.longest_linear_sequence
        total_steps = dag.num_steps
        n_bbs = dag.num_leaves

        # LLS score
        if n_bbs <= 1:
            lls_score = 1.0
        else:
            worst_lls = n_bbs - 1
            lls_score = max(0.0, min(1.0, 1.0 - (lls - 1) / max(worst_lls - 1, 1)))

        # SMARTS validation
        matched_reactions, validation_coverage = _validate_dag_with_smarts(dag, active_lib)

        # Check constraints
        if cfg:
            constraints_satisfied = cfg.check_route_metrics(total_steps, lls)
            if cfg.require_all_validated and validation_coverage < 1.0:
                constraints_satisfied = False

    except Exception as e:
        logger.warning(f"[Scoring] DAG construction failed for {smiles}: {e}")
        convergence = 0.5
        lls_score = 0.5
        lls = num_frags - 1
        total_steps = num_frags - 1
        n_bbs = num_frags
        dag = None

    # Validation score: combination of coverage and avg reliability of matched
    if matched_reactions:
        avg_rel = sum(r for _, r in matched_reactions) / len(matched_reactions)
        validation_score = validation_coverage * avg_rel
    else:
        validation_score = 0.0

    # Penalize if constraints not satisfied
    constraint_multiplier = 1.0 if constraints_satisfied else 0.5

    total = constraint_multiplier * (
        step_weight * step_score
        + availability_weight * availability_score
        + feasibility_weight * bond_feasibility_score
        + convergence_weight * convergence
        + lls_weight * lls_score
        + validation_weight * validation_score
    )

    logger.debug(
        f"[Scoring+SMARTS] {smiles}: steps={step_score:.2f}, avail={availability_score:.2f}, "
        f"bond_feas={bond_feasibility_score:.2f}, conv={convergence:.2f}, "
        f"lls={lls_score:.2f}, valid={validation_score:.2f}, total={total:.2f}"
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
        matched_reactions=matched_reactions,
        validation_coverage=validation_coverage,
        constraints_satisfied=constraints_satisfied,
    )


# ---------------------------------------------------------------------------
# Tier 3: Standalone SMARTS retrosynthesis scoring
# ---------------------------------------------------------------------------


def compute_score_smarts(
    smiles: str,
    compound_filter: CompoundFilter,
    reaction_library: ReactionLibrary,
    *,
    # Simple constraint params
    allowed_reactions: list[str] | None = None,
    blocked_reactions: list[str] | None = None,
    allowed_classes: list[str] | None = None,
    max_steps: int | None = None,
    max_lls: int | None = None,
    min_reliability: float = 0.0,
    # Or config object
    constraints: ConstraintConfig | None = None,
    # Search params
    max_depth: int = 3,
    max_nodes: int = 500,
    max_routes: int = 5,
) -> float:
    """Tier 3: Standalone SMARTS-based retrosynthesis scoring.

    Applies reaction SMARTS in reverse without relying on BRICS fragmentation.
    Covers reactions BRICS cannot (Pictet-Spengler, Fischer indole, Ugi, etc.).

    Uses the CompoundFilter as the purchasability oracle — fragments are
    checked against the BB catalog the same way BRICS solutions are.

    Returns a score in [0, 1] based on: whether the molecule can be solved,
    the average reliability of reactions used, and the route efficiency.

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter for BB checking.
        reaction_library: ReactionLibrary with SMARTS patterns.
        allowed_reactions: Whitelist reaction names.
        blocked_reactions: Blacklist reaction names.
        allowed_classes: Whitelist reaction classes.
        max_steps: Maximum synthesis steps.
        max_lls: Maximum LLS.
        min_reliability: Minimum reaction reliability.
        constraints: Full ConstraintConfig (overrides simple params).
        max_depth: Maximum retrosynthesis tree depth.
        max_nodes: Maximum nodes to explore.
        max_routes: Maximum routes to return.

    Returns:
        Score in [0.0, 1.0].
    """
    from fragmentretro.smarts_retro import SmartsRetrosynthesis

    cfg = build_constraints(
        constraints=constraints,
        allowed_reactions=allowed_reactions,
        blocked_reactions=blocked_reactions,
        allowed_classes=allowed_classes,
        max_steps=max_steps,
        max_lls=max_lls,
        min_reliability=min_reliability,
    )

    def is_purchasable(smi: str) -> bool:
        """Check if a SMILES is in the BB catalog."""
        try:
            return compound_filter.has_match(smi)
        except Exception:
            return False

    retro = SmartsRetrosynthesis(
        library=reaction_library,
        max_depth=max_depth,
        max_nodes=max_nodes,
        constraints=cfg,
    )

    try:
        routes = retro.retrosynthesise(
            smiles,
            is_purchasable=is_purchasable,
            max_routes=max_routes,
        )
    except Exception as e:
        logger.warning(f"[SmartsScoring] Failed for {smiles}: {e}")
        return 0.0

    if not routes:
        return 0.0

    # Pick best route
    best = max(routes, key=lambda r: (r.is_solved, r.avg_reliability))

    if not best.is_solved:
        # Partial credit: fraction of leaves that are BBs
        total_leaves = best.num_leaves
        bb_leaves = sum(1 for _ in _iter_leaves(best) if _.is_building_block)
        partial = bb_leaves / total_leaves if total_leaves > 0 else 0.0
        return 0.3 * partial * best.avg_reliability

    # Full solution: score based on reliability and efficiency
    reliability_score = best.avg_reliability
    steps = best.num_steps
    efficiency_score = 1.0 / (1.0 + 0.2 * steps)  # gentle penalty for more steps

    return 0.5 * reliability_score + 0.5 * efficiency_score


def _iter_leaves(node: object):
    """Iterate over leaf nodes of a RetroSynthNode tree."""
    from fragmentretro.smarts_retro import RetroSynthNode

    n: RetroSynthNode = node  # type: ignore[assignment]
    if n.is_leaf:
        yield n
    else:
        for c in n.children:
            yield from _iter_leaves(c)


# ---------------------------------------------------------------------------
# Unified BRICS tier computation (single _run_retro call)
# ---------------------------------------------------------------------------


@dataclass
class AllBricsTiersResult:
    """Results from a single _run_retro call used to derive all BRICS tiers.

    Attributes:
        binary: T1 binary result (solved / not solved).
        continuous_score: T1 continuous score in [0, 1].
        dag_result: T1 DAG result (DAGScoreResult).
        validated_result: T2 validated result (DAGScoreResult), None if not requested.
    """

    binary: bool
    continuous_score: float
    dag_result: DAGScoreResult
    validated_result: DAGScoreResult | None = None


def compute_all_brics_tiers(
    smiles: str,
    compound_filter: CompoundFilter,
    reaction_library: ReactionLibrary | None = None,
    *,
    # Which tiers to compute
    include_binary: bool = True,
    include_continuous: bool = True,
    include_dag: bool = True,
    include_validated: bool = False,
    # Shared retro params
    use_rbrics: bool = True,
    solution_cap: int = 5,
    # Continuous scoring weights
    cont_step_weight: float = 0.35,
    cont_availability_weight: float = 0.30,
    cont_feasibility_weight: float = 0.35,
    # DAG scoring weights
    dag_step_weight: float = 0.25,
    dag_availability_weight: float = 0.20,
    dag_feasibility_weight: float = 0.25,
    dag_convergence_weight: float = 0.15,
    dag_lls_weight: float = 0.15,
    # Validated scoring weights
    val_step_weight: float = 0.20,
    val_availability_weight: float = 0.15,
    val_feasibility_weight: float = 0.25,
    val_convergence_weight: float = 0.10,
    val_lls_weight: float = 0.10,
    val_validation_weight: float = 0.20,
    # Constraint params for validated tier
    constraints: ConstraintConfig | None = None,
    allowed_reactions: list[str] | None = None,
    blocked_reactions: list[str] | None = None,
    allowed_classes: list[str] | None = None,
    max_steps: int | None = None,
    max_lls: int | None = None,
    min_reliability: float = 0.0,
    require_all_validated: bool = False,
) -> AllBricsTiersResult:
    """Compute all BRICS-based tier scores from a single retrosynthesis run.

    This avoids redundant calls to ``_run_retro`` when multiple tiers are
    needed for the same molecule. The expensive BRICS fragmentation and
    substructure matching against the BB catalog is performed **once**,
    then each requested tier score is derived from the shared result.

    Args:
        smiles: Target molecule SMILES.
        compound_filter: Pre-loaded CompoundFilter (shared singleton).
        reaction_library: ReactionLibrary for T2 validation (optional).
        include_binary .. include_validated: Which tiers to compute.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        solution_cap: Max solutions to enumerate.
        cont_*/dag_*/val_*: Per-tier scoring weights.
        constraints .. require_all_validated: Constraint params for T2.

    Returns:
        AllBricsTiersResult with all requested tier scores.
    """
    from fragmentretro.retro_dag import build_best_dag

    # Determine binary_mode: only use binary if that's the ONLY tier requested
    only_binary = include_binary and not (include_continuous or include_dag or include_validated)

    # --- Single _run_retro call ---
    try:
        retro_tool, retro_solution = _run_retro(
            smiles,
            compound_filter,
            binary_mode=only_binary,
            use_rbrics=use_rbrics,
            solution_cap=1 if only_binary else solution_cap,
        )
    except Exception as e:
        logger.warning(f"[Scoring] Failed to score {smiles}: {e}")
        return AllBricsTiersResult(
            binary=False,
            continuous_score=0.0,
            dag_result=DAGScoreResult(score=0.0, feasible=False),
            validated_result=DAGScoreResult(score=0.0, feasible=False) if include_validated else None,
        )

    has_solutions = len(retro_solution.solutions) > 0

    # --- T1 binary ---
    binary_result = has_solutions

    if not has_solutions:
        return AllBricsTiersResult(
            binary=False,
            continuous_score=0.0,
            dag_result=DAGScoreResult(score=0.0, feasible=False),
            validated_result=DAGScoreResult(score=0.0, feasible=False) if include_validated else None,
        )

    # --- Shared computation for continuous / DAG / validated ---
    best_solution: SolutionType = min(retro_solution.solutions, key=len)
    num_frags = len(best_solution)
    max_frags = retro_tool.num_fragments

    # Step score (shared)
    step_score = 1.0 if max_frags <= 1 else 1.0 - (num_frags - 1) / (max_frags - 1)

    # BB availability (shared)
    bb_counts = []
    for comb in best_solution:
        bbs = retro_tool.comb_bbs_dict.get(comb, set())
        bb_counts.append(len(bbs))
    min_bb_count = min(bb_counts) if bb_counts else 0
    availability_score = min(1.0, math.log1p(min_bb_count) / math.log1p(100))

    # Bond feasibility (shared)
    bond_types = retro_tool.fragmenter.get_bond_types_for_solution(best_solution)
    bond_feasibility_score = get_solution_bond_feasibility(bond_types)

    # --- T1 continuous ---
    continuous_score = (
        cont_step_weight * step_score
        + cont_availability_weight * availability_score
        + cont_feasibility_weight * bond_feasibility_score
    )

    # --- DAG construction (shared by T1_dag and T2_validated) ---
    dag = None
    convergence = 0.5
    lls = num_frags - 1
    total_steps = num_frags - 1
    n_bbs = num_frags
    lls_score = 0.5

    if include_dag or include_validated:
        try:
            dag = build_best_dag(best_solution, retro_tool.fragmenter, retro_tool.comb_bbs_dict)
            convergence = dag.convergence_score
            lls = dag.longest_linear_sequence
            total_steps = dag.num_steps
            n_bbs = dag.num_leaves

            if n_bbs <= 1:
                lls_score = 1.0
            else:
                worst_lls = n_bbs - 1
                lls_score = max(0.0, min(1.0, 1.0 - (lls - 1) / max(worst_lls - 1, 1)))
        except Exception as e:
            logger.warning(f"[Scoring] DAG construction failed for {smiles}: {e}")

    # --- T1 DAG score ---
    dag_total = (
        dag_step_weight * step_score
        + dag_availability_weight * availability_score
        + dag_feasibility_weight * bond_feasibility_score
        + dag_convergence_weight * convergence
        + dag_lls_weight * lls_score
    )

    dag_result = DAGScoreResult(
        score=dag_total,
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

    # --- T2 validated score ---
    validated_result: DAGScoreResult | None = None
    if include_validated:
        matched_reactions: list[tuple[str, float]] = []
        validation_coverage = 0.0
        constraints_satisfied = True

        cfg = build_constraints(
            constraints=constraints,
            allowed_reactions=allowed_reactions,
            blocked_reactions=blocked_reactions,
            allowed_classes=allowed_classes,
            max_steps=max_steps,
            max_lls=max_lls,
            min_reliability=min_reliability,
            require_all_validated=require_all_validated,
        )

        active_lib = reaction_library
        if cfg and reaction_library is not None:
            active_lib = cfg.get_filtered_library(reaction_library)

        if dag is not None and active_lib is not None:
            matched_reactions, validation_coverage = _validate_dag_with_smarts(dag, active_lib)

            if cfg:
                constraints_satisfied = cfg.check_route_metrics(total_steps, lls)
                if cfg.require_all_validated and validation_coverage < 1.0:
                    constraints_satisfied = False

        # Validation score
        if matched_reactions:
            avg_rel = sum(r for _, r in matched_reactions) / len(matched_reactions)
            validation_score = validation_coverage * avg_rel
        else:
            validation_score = 0.0

        constraint_multiplier = 1.0 if constraints_satisfied else 0.5

        val_total = constraint_multiplier * (
            val_step_weight * step_score
            + val_availability_weight * availability_score
            + val_feasibility_weight * bond_feasibility_score
            + val_convergence_weight * convergence
            + val_lls_weight * lls_score
            + val_validation_weight * validation_score
        )

        validated_result = DAGScoreResult(
            score=val_total,
            feasible=True,
            total_steps=total_steps,
            longest_linear_sequence=lls,
            num_building_blocks=n_bbs,
            convergence_score=convergence,
            step_score=step_score,
            availability_score=availability_score,
            bond_feasibility_score=bond_feasibility_score,
            dag=dag,
            matched_reactions=matched_reactions,
            validation_coverage=validation_coverage,
            constraints_satisfied=constraints_satisfied,
        )

    return AllBricsTiersResult(
        binary=binary_result,
        continuous_score=continuous_score,
        dag_result=dag_result,
        validated_result=validated_result,
    )
