"""Bond feasibility scoring based on BRICS environment labels.

Maps BRICS bond type pairs to feasibility scores reflecting how reliably
the corresponding reconnection chemistry works in practice. Scores range
from 0.0 (unreliable/rare) to 1.0 (routine/high-yielding).
"""

# Feasibility scores for BRICS bond reconnections.
# Keys are frozensets of (env_label_1, env_label_2) so order doesn't matter.
# Scores are heuristic estimates based on common reaction reliability:
#   1.0 = routine (amide coupling, ether formation, sulfonamide, Suzuki)
#   0.8 = reliable but may need optimization (reductive amination, C-N coupling)
#   0.6 = moderate (C-C cross-coupling requiring specific catalysts)
#   0.4 = challenging (ring-opening, unusual bond formations)
#   0.2 = difficult (non-standard reconnections)

BOND_FEASIBILITY: dict[frozenset[str], float] = {
    # L1 (C=O) connections — amide/ester formation
    frozenset({"1", "3"}): 1.0,    # ester formation
    frozenset({"1", "5"}): 1.0,    # amide coupling
    frozenset({"1", "10"}): 0.8,   # lactam
    # L3 (O) connections — ether/ester
    frozenset({"3", "4"}): 0.9,    # ether (Williamson)
    frozenset({"3", "13"}): 0.8,
    frozenset({"3", "14"}): 0.8,   # aryl ether (SNAr/Buchwald)
    frozenset({"3", "14b"}): 0.8,
    frozenset({"3", "15"}): 0.8,
    frozenset({"3", "16"}): 0.8,
    frozenset({"3", "16b"}): 0.8,
    # L4 (C-C acyclic) connections
    frozenset({"4", "5"}): 0.8,    # reductive amination
    frozenset({"4", "11"}): 0.7,   # C-S bond
    # L5 (N) connections — amination/sulfonamide
    frozenset({"5", "12"}): 1.0,   # sulfonamide
    frozenset({"5", "12b"}): 1.0,
    frozenset({"5", "13"}): 0.8,
    frozenset({"5", "14"}): 0.8,   # Buchwald-Hartwig
    frozenset({"5", "14b"}): 0.8,
    frozenset({"5", "15"}): 0.7,
    frozenset({"5", "16"}): 0.8,
    frozenset({"5", "16b"}): 0.8,
    # L51 (N-N, N-O)
    frozenset({"51", "1"}): 0.7,
    frozenset({"51", "4"}): 0.6,
    frozenset({"51", "12"}): 0.8,
    frozenset({"51", "12b"}): 0.8,
    frozenset({"51", "13"}): 0.6,
    frozenset({"51", "14"}): 0.6,
    frozenset({"51", "14b"}): 0.6,
    frozenset({"51", "15"}): 0.5,
    frozenset({"51", "16"}): 0.6,
    frozenset({"51", "16b"}): 0.6,
    # L6 (C(=O) acyclic)
    frozenset({"6", "13"}): 0.7,
    frozenset({"6", "14"}): 0.7,
    frozenset({"6", "15"}): 0.6,
    frozenset({"6", "16"}): 0.8,   # Friedel-Crafts-like
    # L7 (C=C)
    frozenset({"7", "7"}): 0.6,    # olefin metathesis / Wittig
    frozenset({"7a", "7b"}): 0.6,
    # L8 (C-ring) connections — cross-coupling
    frozenset({"8", "9"}): 0.5,
    frozenset({"8", "10"}): 0.5,
    frozenset({"8", "13"}): 0.6,
    frozenset({"8", "14"}): 0.6,
    frozenset({"8", "14b"}): 0.6,
    frozenset({"8", "15"}): 0.6,
    frozenset({"8", "16"}): 0.6,
    frozenset({"8", "16b"}): 0.6,
    # L81 (amidine)
    frozenset({"81", "8"}): 0.4,
    frozenset({"81", "9"}): 0.4,
    frozenset({"81", "10"}): 0.4,
    frozenset({"81", "13"}): 0.4,
    frozenset({"81", "14"}): 0.4,
    frozenset({"81", "14b"}): 0.4,
    frozenset({"81", "15"}): 0.4,
    frozenset({"81", "16"}): 0.4,
    frozenset({"81", "16b"}): 0.4,
    # L9 (ring N)
    frozenset({"9", "13"}): 0.5,
    frozenset({"9", "14"}): 0.5,
    frozenset({"9", "14b"}): 0.5,
    frozenset({"9", "15"}): 0.5,
    frozenset({"9", "16"}): 0.5,
    frozenset({"9", "16b"}): 0.5,
    # L10 (ring N-C=O)
    frozenset({"10", "13"}): 0.5,
    frozenset({"10", "14"}): 0.5,
    frozenset({"10", "14b"}): 0.5,
    frozenset({"10", "15"}): 0.5,
    frozenset({"10", "16"}): 0.5,
    frozenset({"10", "16b"}): 0.5,
    # L11 (S)
    frozenset({"11", "13"}): 0.6,
    frozenset({"11", "14"}): 0.6,
    frozenset({"11", "15"}): 0.6,
    frozenset({"11", "16"}): 0.6,
    # L12b (SO2)
    frozenset({"12b", "12b"}): 0.4,
    frozenset({"12b", "4"}): 0.6,
    frozenset({"12b", "5"}): 0.9,
    frozenset({"12b", "13"}): 0.6,
    frozenset({"12b", "14"}): 0.6,
    frozenset({"12b", "14b"}): 0.6,
    frozenset({"12b", "15"}): 0.6,
    frozenset({"12b", "16"}): 0.6,
    frozenset({"12b", "16b"}): 0.6,
    # Ring C-C connections (Suzuki, Heck, etc.)
    frozenset({"13", "14"}): 0.7,
    frozenset({"13", "14b"}): 0.7,
    frozenset({"13", "15"}): 0.6,
    frozenset({"13", "16"}): 0.7,
    frozenset({"14", "14"}): 0.8,   # biaryl Suzuki
    frozenset({"14", "14b"}): 0.8,
    frozenset({"14", "15"}): 0.7,
    frozenset({"14", "16"}): 0.8,   # Suzuki
    frozenset({"14b", "14b"}): 0.7,
    frozenset({"14b", "15"}): 0.6,
    frozenset({"14b", "16"}): 0.7,
    frozenset({"14b", "16b"}): 0.7,
    frozenset({"15", "16"}): 0.7,
    frozenset({"15", "16b"}): 0.6,
    frozenset({"16", "16"}): 0.8,   # biaryl
    frozenset({"16", "16b"}): 0.7,
    frozenset({"16b", "16b"}): 0.6,
    # L17 (isobutyl/isopropyl)
    frozenset({"17", "5"}): 0.6,
    frozenset({"17", "8"}): 0.4,
    frozenset({"17", "9"}): 0.4,
    frozenset({"17", "10"}): 0.4,
    frozenset({"17", "11"}): 0.5,
    frozenset({"17", "12"}): 0.5,
    frozenset({"17", "12b"}): 0.5,
    frozenset({"17", "13"}): 0.5,
    frozenset({"17", "14"}): 0.5,
    frozenset({"17", "14b"}): 0.5,
    frozenset({"17", "15"}): 0.5,
    frozenset({"17", "16"}): 0.5,
    frozenset({"17", "16b"}): 0.5,
    frozenset({"17", "17"}): 0.3,
    frozenset({"17", "51"}): 0.5,
    frozenset({"17", "81"}): 0.3,
    # L18/L19 ring fusions
    frozenset({"18", "19"}): 0.3,   # ring fusion — challenging
    frozenset({"182", "192"}): 0.2, # double bond ring fusion
    # Aliphatic chain breaks
    frozenset({"20", "21"}): 0.7,   # simple C-C
    frozenset({"22", "23"}): 0.6,
    # L30 (C#N, C#C)
    frozenset({"30", "30"}): 0.5,
    frozenset({"30", "4"}): 0.6,
    frozenset({"30", "5"}): 0.6,
    frozenset({"30", "6"}): 0.5,
    frozenset({"30", "8"}): 0.5,
    frozenset({"30", "9"}): 0.5,
    frozenset({"30", "10"}): 0.5,
    frozenset({"30", "11"}): 0.5,
    frozenset({"30", "12"}): 0.5,
    frozenset({"30", "12b"}): 0.5,
    frozenset({"30", "13"}): 0.5,
    frozenset({"30", "14"}): 0.5,
    frozenset({"30", "14b"}): 0.5,
    frozenset({"30", "15"}): 0.5,
    frozenset({"30", "16"}): 0.5,
    frozenset({"30", "16b"}): 0.5,
    frozenset({"30", "51"}): 0.5,
    frozenset({"30", "81"}): 0.4,
}

# Default score for unknown bond type pairs
DEFAULT_FEASIBILITY = 0.3


def get_bond_feasibility(env_label_1: str, env_label_2: str) -> float:
    """Get the feasibility score for a BRICS bond reconnection.

    Args:
        env_label_1: First BRICS environment label (e.g., '1', '5', '16').
        env_label_2: Second BRICS environment label.

    Returns:
        Feasibility score between 0.0 and 1.0.
    """
    key = frozenset({env_label_1, env_label_2})
    return BOND_FEASIBILITY.get(key, DEFAULT_FEASIBILITY)


def get_solution_bond_feasibility(bond_types: list[tuple[str, str]]) -> float:
    """Compute average bond feasibility for a list of bond types in a solution.

    Args:
        bond_types: List of (env_label_1, env_label_2) tuples from fragmenter.

    Returns:
        Average feasibility score, or 1.0 if no bonds (single fragment = target is a BB).
    """
    if not bond_types:
        return 1.0
    scores = [get_bond_feasibility(t1, t2) for t1, t2 in bond_types]
    return sum(scores) / len(scores)
