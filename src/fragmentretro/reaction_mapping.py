"""Mapping from BRICS bond-type pairs to named synthetic reaction classes.

Each BRICS disconnection corresponds to a known reaction in the forward
(synthetic) direction. This module provides human-readable reaction names
and the forward reaction description for each bond type pair.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionInfo:
    """Information about a synthetic reaction corresponding to a BRICS bond."""

    name: str
    description: str
    forward_class: str  # broad reaction class


# Mapping from frozenset({env1, env2}) -> ReactionInfo
# Covers the most common BRICS reconnection types.
REACTION_MAP: dict[frozenset[str], ReactionInfo] = {
    # L1-L3: ester bond
    frozenset({"1", "3"}): ReactionInfo(
        name="Ester formation",
        description="Acyl chloride/acid + alcohol → ester",
        forward_class="acylation",
    ),
    # L1-L5: amide bond
    frozenset({"1", "5"}): ReactionInfo(
        name="Amide coupling",
        description="Carboxylic acid + amine → amide (e.g. HATU, EDC)",
        forward_class="acylation",
    ),
    # L1-L10: lactam
    frozenset({"1", "10"}): ReactionInfo(
        name="Lactam formation",
        description="Intramolecular amide cyclization",
        forward_class="acylation",
    ),
    # L3-L4: ether
    frozenset({"3", "4"}): ReactionInfo(
        name="Williamson ether synthesis",
        description="Alkoxide + alkyl halide → ether",
        forward_class="substitution",
    ),
    frozenset({"3", "13"}): ReactionInfo(
        name="Ether formation (ring C-O)",
        description="Alcohol + ring carbon → ether",
        forward_class="substitution",
    ),
    frozenset({"3", "14"}): ReactionInfo(
        name="Aryl ether formation",
        description="SNAr or Buchwald C-O coupling",
        forward_class="cross-coupling",
    ),
    frozenset({"3", "14b"}): ReactionInfo(
        name="Aryl ether formation",
        description="SNAr or Buchwald C-O coupling",
        forward_class="cross-coupling",
    ),
    frozenset({"3", "15"}): ReactionInfo(
        name="Ether formation (C-C ring)",
        description="Alcohol + cyclic carbon → ether",
        forward_class="substitution",
    ),
    frozenset({"3", "16"}): ReactionInfo(
        name="Aryl ether formation",
        description="Phenol + aryl halide → diaryl ether",
        forward_class="cross-coupling",
    ),
    frozenset({"3", "16b"}): ReactionInfo(
        name="Aryl ether formation",
        description="Phenol + aryl halide → diaryl ether",
        forward_class="cross-coupling",
    ),
    # L4-L5: reductive amination
    frozenset({"4", "5"}): ReactionInfo(
        name="Reductive amination",
        description="Aldehyde/ketone + amine → amine (NaBH3CN)",
        forward_class="reductive amination",
    ),
    # L4-L11: thioether
    frozenset({"4", "11"}): ReactionInfo(
        name="Thioether formation",
        description="Alkyl halide + thiol → thioether",
        forward_class="substitution",
    ),
    # L5-L12/L12b: sulfonamide
    frozenset({"5", "12"}): ReactionInfo(
        name="Sulfonamide formation",
        description="Sulfonyl chloride + amine → sulfonamide",
        forward_class="sulfonylation",
    ),
    frozenset({"5", "12b"}): ReactionInfo(
        name="Sulfonamide formation",
        description="Sulfonyl chloride + amine → sulfonamide",
        forward_class="sulfonylation",
    ),
    # L5-L14/L16: Buchwald-Hartwig
    frozenset({"5", "14"}): ReactionInfo(
        name="Buchwald-Hartwig amination",
        description="Aryl halide + amine → aryl amine (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"5", "14b"}): ReactionInfo(
        name="Buchwald-Hartwig amination",
        description="Aryl halide + amine → aryl amine (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"5", "16"}): ReactionInfo(
        name="Buchwald-Hartwig amination",
        description="Aryl halide + amine → aryl amine (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"5", "16b"}): ReactionInfo(
        name="Buchwald-Hartwig amination",
        description="Aryl halide + amine → aryl amine (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"5", "13"}): ReactionInfo(
        name="N-alkylation",
        description="Amine + ring-C halide → N-alkylated product",
        forward_class="substitution",
    ),
    frozenset({"5", "15"}): ReactionInfo(
        name="N-alkylation",
        description="Amine + cycloalkyl halide → N-alkylated product",
        forward_class="substitution",
    ),
    # L6: acyl connections
    frozenset({"6", "13"}): ReactionInfo(
        name="Friedel-Crafts acylation",
        description="Acyl halide + ring → ketone",
        forward_class="acylation",
    ),
    frozenset({"6", "14"}): ReactionInfo(
        name="Friedel-Crafts acylation",
        description="Acyl halide + arene → aryl ketone",
        forward_class="acylation",
    ),
    frozenset({"6", "15"}): ReactionInfo(
        name="Acylation",
        description="Acyl chloride + C-ring → ketone",
        forward_class="acylation",
    ),
    frozenset({"6", "16"}): ReactionInfo(
        name="Friedel-Crafts acylation",
        description="Acyl halide + arene → aryl ketone",
        forward_class="acylation",
    ),
    # L7: olefin
    frozenset({"7a", "7b"}): ReactionInfo(
        name="Olefination",
        description="Wittig / HWE / olefin metathesis",
        forward_class="olefination",
    ),
    # L8/L81: C-ring cross-coupling
    frozenset({"8", "14"}): ReactionInfo(
        name="C-C cross-coupling",
        description="Alkyl-aryl coupling (Suzuki, Negishi)",
        forward_class="cross-coupling",
    ),
    frozenset({"8", "16"}): ReactionInfo(
        name="C-C cross-coupling",
        description="Alkyl-aryl coupling (Suzuki, Negishi)",
        forward_class="cross-coupling",
    ),
    # Aryl-aryl (Suzuki, etc.)
    frozenset({"14", "14"}): ReactionInfo(
        name="Suzuki coupling",
        description="Aryl boronic acid + aryl halide → biaryl (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"14", "16"}): ReactionInfo(
        name="Suzuki coupling",
        description="Aryl boronic acid + aryl halide → biaryl (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"16", "16"}): ReactionInfo(
        name="Suzuki coupling",
        description="Aryl boronic acid + aryl halide → biaryl (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"14", "14b"}): ReactionInfo(
        name="Suzuki coupling",
        description="Aryl boronic acid + aryl halide → biaryl (Pd catalyst)",
        forward_class="cross-coupling",
    ),
    frozenset({"14b", "16"}): ReactionInfo(
        name="Suzuki coupling",
        description="Aryl/ring boronic acid + aryl halide → biaryl",
        forward_class="cross-coupling",
    ),
    frozenset({"13", "16"}): ReactionInfo(
        name="C-C cross-coupling",
        description="Ring C + aryl halide coupling",
        forward_class="cross-coupling",
    ),
    frozenset({"15", "16"}): ReactionInfo(
        name="C-C cross-coupling",
        description="Cycloalkyl + aryl coupling",
        forward_class="cross-coupling",
    ),
    # Ring fusions
    frozenset({"18", "19"}): ReactionInfo(
        name="Ring fusion",
        description="Annulation / ring-closing reaction",
        forward_class="cyclization",
    ),
    frozenset({"182", "192"}): ReactionInfo(
        name="Ring fusion (double bond)",
        description="Ring-closing metathesis or annulation",
        forward_class="cyclization",
    ),
    # Aliphatic chain
    frozenset({"20", "21"}): ReactionInfo(
        name="C-C bond formation",
        description="Alkyl chain coupling (Grignard, Suzuki sp3)",
        forward_class="C-C formation",
    ),
    frozenset({"22", "23"}): ReactionInfo(
        name="C-C bond formation",
        description="Long-chain coupling",
        forward_class="C-C formation",
    ),
}

# Default for unmapped bond types
DEFAULT_REACTION = ReactionInfo(
    name="Unknown disconnection",
    description="BRICS bond type not mapped to a specific reaction",
    forward_class="unknown",
)


def get_reaction_info(env_label_1: str, env_label_2: str) -> ReactionInfo:
    """Look up the synthetic reaction for a BRICS bond type pair.

    Args:
        env_label_1: First BRICS environment label.
        env_label_2: Second BRICS environment label.

    Returns:
        ReactionInfo with name, description, and forward_class.
    """
    key = frozenset({env_label_1, env_label_2})
    return REACTION_MAP.get(key, DEFAULT_REACTION)
