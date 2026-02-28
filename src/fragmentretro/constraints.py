"""Constraint system for reaction-aware retrosynthesis.

Provides both simple parameter-based filtering and a rich ConstraintConfig
object for complex constraint specifications.

Simple usage (kwargs in scoring/DAG functions):
    score = compute_score(smiles, cf, allowed_reactions=["Suzuki"], max_steps=3)

Complex usage (ConstraintConfig):
    config = ConstraintConfig(
        allowed_reactions=["Suzuki", "amide_coupling"],
        blocked_reactions=["Grignard"],
        allowed_classes=["coupling"],
        max_steps=3,
        max_lls=2,
        min_reliability=0.8,
        require_all_validated=True,
    )
    score = compute_score(smiles, cf, constraints=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fragmentretro.reaction_library import Reaction, ReactionLibrary


@dataclass
class ConstraintConfig:
    """Rich constraint specification for retrosynthetic route filtering.

    Attributes:
        allowed_reactions: Whitelist of reaction names (partial match, case-insensitive).
            If set, only these reactions are permitted.
        blocked_reactions: Blacklist of reaction names. Overrides allowed_reactions.
        allowed_classes: Whitelist of reaction classes (e.g., 'coupling', 'ring_closure').
        blocked_classes: Blacklist of reaction classes.
        max_steps: Maximum total synthetic steps allowed.
        max_lls: Maximum longest linear sequence allowed.
        min_reliability: Minimum per-reaction reliability score.
        max_reactants: Maximum reactants per step (e.g., 2 excludes multicomponent).
        require_all_validated: If True, every disconnection must match a library reaction.
            If False (default), unmatched disconnections use heuristic scoring.
    """

    allowed_reactions: list[str] | None = None
    blocked_reactions: list[str] | None = None
    allowed_classes: list[str] | None = None
    blocked_classes: list[str] | None = None
    max_steps: int | None = None
    max_lls: int | None = None
    min_reliability: float = 0.0
    max_reactants: int | None = None
    require_all_validated: bool = False

    def get_filtered_library(self, library: ReactionLibrary) -> ReactionLibrary:
        """Apply this constraint's filters to a reaction library.

        Returns a new library containing only permitted reactions.
        """
        filtered = library

        # Apply class filters
        if self.allowed_classes:
            filtered = filtered.filter(classes=self.allowed_classes)
        if self.blocked_classes:
            # Remove blocked classes by filtering to non-blocked
            all_classes = filtered.reaction_classes
            keep = [c for c in all_classes if c not in self.blocked_classes]
            if keep:
                filtered = filtered.filter(classes=keep)

        # Apply name filters
        if self.allowed_reactions:
            filtered = filtered.filter(names=self.allowed_reactions)
        if self.blocked_reactions:
            blocked_lower = [n.lower() for n in self.blocked_reactions]
            keep_rxns = [
                r for r in filtered.reactions
                if not any(b in r.name.lower() for b in blocked_lower)
            ]
            filtered = ReactionLibrary(keep_rxns)

        # Apply reliability and reactant filters
        if self.min_reliability > 0 or self.max_reactants is not None:
            filtered = filtered.filter(
                min_reliability=self.min_reliability,
                max_reactants=self.max_reactants,
            )

        return filtered

    def check_route_metrics(
        self,
        total_steps: int,
        lls: int,
    ) -> bool:
        """Check whether a route's metrics satisfy step/LLS constraints.

        Args:
            total_steps: Number of synthetic steps in the route.
            lls: Longest linear sequence of the route.

        Returns:
            True if the route passes all metric constraints.
        """
        if self.max_steps is not None and total_steps > self.max_steps:
            return False
        if self.max_lls is not None and lls > self.max_lls:
            return False
        return True

    def is_reaction_allowed(self, reaction: Reaction) -> bool:
        """Check whether a specific reaction is permitted under these constraints."""
        # Blocked reactions override everything
        if self.blocked_reactions:
            blocked_lower = [n.lower() for n in self.blocked_reactions]
            if any(b in reaction.name.lower() for b in blocked_lower):
                return False
        if self.blocked_classes:
            if reaction.reaction_class in self.blocked_classes:
                return False

        # Allowed reactions whitelist
        if self.allowed_reactions:
            allowed_lower = [n.lower() for n in self.allowed_reactions]
            if not any(a in reaction.name.lower() for a in allowed_lower):
                return False
        if self.allowed_classes:
            if reaction.reaction_class not in self.allowed_classes:
                return False

        if reaction.reliability < self.min_reliability:
            return False
        if self.max_reactants is not None and reaction.num_reactants > self.max_reactants:
            return False

        return True


def build_constraints(
    *,
    constraints: ConstraintConfig | None = None,
    allowed_reactions: list[str] | None = None,
    blocked_reactions: list[str] | None = None,
    allowed_classes: list[str] | None = None,
    max_steps: int | None = None,
    max_lls: int | None = None,
    min_reliability: float = 0.0,
    require_all_validated: bool = False,
) -> ConstraintConfig | None:
    """Build a ConstraintConfig from either a config object or simple kwargs.

    If a ConstraintConfig is provided, it takes precedence.
    If simple kwargs are provided, they're merged into a new config.
    If neither is provided, returns None (no constraints).

    This is the bridge between the simple API and the config object API.
    """
    if constraints is not None:
        return constraints

    # Check if any simple params were provided
    has_simple = any([
        allowed_reactions, blocked_reactions, allowed_classes,
        max_steps is not None, max_lls is not None,
        min_reliability > 0, require_all_validated,
    ])

    if not has_simple:
        return None

    return ConstraintConfig(
        allowed_reactions=allowed_reactions,
        blocked_reactions=blocked_reactions,
        allowed_classes=allowed_classes,
        max_steps=max_steps,
        max_lls=max_lls,
        min_reliability=min_reliability,
        require_all_validated=require_all_validated,
    )
