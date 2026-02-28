"""Batch scoring utilities for Reinvent4 RL integration.

Parallelizes scoring across molecules (not within molecules),
which is more efficient for RL batch evaluation.
"""

import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from fragmentretro.scoring import compute_score, is_feasible, load_compound_filter, compute_score_with_dag, DAGScoreResult
from fragmentretro.utils.filter_compound import CompoundFilter
from fragmentretro.utils.logging_config import logger


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _TimeoutError("Scoring timed out")


def _score_single_binary(
    smiles: str,
    mol_properties_path: str,
    fpSize: int,
    use_rbrics: bool,
) -> float:
    """Worker function for binary scoring (runs in subprocess)."""
    try:
        cf = load_compound_filter(mol_properties_path, fpSize)
        return 1.0 if is_feasible(smiles, cf, use_rbrics=use_rbrics) else 0.0
    except Exception as e:
        logger.warning(f"[Batch] Binary scoring failed for {smiles}: {e}")
        return 0.0


def _score_single_continuous(
    smiles: str,
    mol_properties_path: str,
    fpSize: int,
    use_rbrics: bool,
    solution_cap: int,
) -> float:
    """Worker function for continuous scoring (runs in subprocess)."""
    try:
        cf = load_compound_filter(mol_properties_path, fpSize)
        return compute_score(smiles, cf, use_rbrics=use_rbrics, solution_cap=solution_cap)
    except Exception as e:
        logger.warning(f"[Batch] Continuous scoring failed for {smiles}: {e}")
        return 0.0


def score_batch_binary(
    smiles_list: list[str],
    mol_properties_path: str | Path,
    max_workers: int = 4,
    timeout_per_mol: float = 10.0,
    use_rbrics: bool = True,
    fpSize: int = 2048,
) -> list[float]:
    """Score a batch of molecules with binary feasibility (0.0 or 1.0).

    Parallelizes across molecules. Each molecule gets its own timeout.

    Args:
        smiles_list: List of target SMILES strings.
        mol_properties_path: Path to precomputed BB properties JSON.
        max_workers: Number of parallel workers.
        timeout_per_mol: Timeout in seconds per molecule.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        fpSize: Fingerprint size.

    Returns:
        List of scores (0.0 or 1.0) in same order as input.
    """
    path_str = str(mol_properties_path)
    scores = [0.0] * len(smiles_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            i: executor.submit(_score_single_binary, smi, path_str, fpSize, use_rbrics)
            for i, smi in enumerate(smiles_list)
        }
        for i, future in futures.items():
            try:
                scores[i] = future.result(timeout=timeout_per_mol)
            except (FuturesTimeoutError, Exception) as e:
                logger.warning(f"[Batch] Timeout/error for molecule {i}: {e}")
                scores[i] = 0.0

    return scores


def score_batch_continuous(
    smiles_list: list[str],
    mol_properties_path: str | Path,
    max_workers: int = 4,
    timeout_per_mol: float = 30.0,
    use_rbrics: bool = True,
    solution_cap: int = 5,
    fpSize: int = 2048,
) -> list[float]:
    """Score a batch of molecules with continuous feasibility scores [0, 1].

    Args:
        smiles_list: List of target SMILES strings.
        mol_properties_path: Path to precomputed BB properties JSON.
        max_workers: Number of parallel workers.
        timeout_per_mol: Timeout in seconds per molecule.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        solution_cap: Max solutions to enumerate per molecule.
        fpSize: Fingerprint size.

    Returns:
        List of scores in [0.0, 1.0] in same order as input.
    """
    path_str = str(mol_properties_path)
    scores = [0.0] * len(smiles_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            i: executor.submit(_score_single_continuous, smi, path_str, fpSize, use_rbrics, solution_cap)
            for i, smi in enumerate(smiles_list)
        }
        for i, future in futures.items():
            try:
                scores[i] = future.result(timeout=timeout_per_mol)
            except (FuturesTimeoutError, Exception) as e:
                logger.warning(f"[Batch] Timeout/error for molecule {i}: {e}")
                scores[i] = 0.0

    return scores


def _score_single_dag(
    smiles: str,
    mol_properties_path: str,
    fpSize: int,
    use_rbrics: bool,
    solution_cap: int,
) -> dict:
    """Worker function for DAG scoring (runs in subprocess). Returns serializable dict."""
    try:
        cf = load_compound_filter(mol_properties_path, fpSize)
        result = compute_score_with_dag(smiles, cf, use_rbrics=use_rbrics, solution_cap=solution_cap)
        return {
            "score": result.score,
            "feasible": result.feasible,
            "total_steps": result.total_steps,
            "longest_linear_sequence": result.longest_linear_sequence,
            "num_building_blocks": result.num_building_blocks,
            "convergence_score": result.convergence_score,
        }
    except Exception as e:
        logger.warning(f"[Batch] DAG scoring failed for {smiles}: {e}")
        return {"score": 0.0, "feasible": False, "total_steps": 0,
                "longest_linear_sequence": 0, "num_building_blocks": 0,
                "convergence_score": 0.0}


def score_batch_with_dag(
    smiles_list: list[str],
    mol_properties_path: str | Path,
    max_workers: int = 4,
    timeout_per_mol: float = 60.0,
    use_rbrics: bool = True,
    solution_cap: int = 5,
    fpSize: int = 2048,
) -> list[dict]:
    """Score a batch with DAG analysis. Returns list of dicts with score + route metrics.

    More expensive than score_batch_continuous - use when you need convergence
    and LLS information.

    Args:
        smiles_list: List of target SMILES strings.
        mol_properties_path: Path to precomputed BB properties JSON.
        max_workers: Number of parallel workers.
        timeout_per_mol: Timeout in seconds per molecule.
        use_rbrics: Use r-BRICS (recommended) or BRICS.
        solution_cap: Max solutions to enumerate per molecule.
        fpSize: Fingerprint size.

    Returns:
        List of dicts with keys: score, feasible, total_steps,
        longest_linear_sequence, num_building_blocks, convergence_score.
    """
    path_str = str(mol_properties_path)
    empty = {"score": 0.0, "feasible": False, "total_steps": 0,
             "longest_linear_sequence": 0, "num_building_blocks": 0,
             "convergence_score": 0.0}
    results: list[dict] = [empty.copy() for _ in smiles_list]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            i: executor.submit(_score_single_dag, smi, path_str, fpSize, use_rbrics, solution_cap)
            for i, smi in enumerate(smiles_list)
        }
        for i, future in futures.items():
            try:
                results[i] = future.result(timeout=timeout_per_mol)
            except (FuturesTimeoutError, Exception) as e:
                logger.warning(f"[Batch] Timeout/error for molecule {i}: {e}")

    return results
