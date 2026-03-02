"""Standalone SMARTS-based retrosynthesis engine.

Applies reaction SMARTS in reverse (product → reactants) to decompose a target
molecule into purchasable building blocks. Covers reactions that BRICS cannot
represent (Pictet-Spengler, Fischer indole, click chemistry, multicomponent, etc.).

Complexity: O(b^h) where b = branching factor (reactions × sites), h = tree depth.
Use max_depth and max_nodes to bound search.

Usage:
    from fragmentretro.reaction_library import ReactionLibrary
    from fragmentretro.smarts_retro import SmartsRetrosynthesis

    lib = ReactionLibrary.default()
    retro = SmartsRetrosynthesis(lib, max_depth=3)
    trees = retro.retrosynthesise("c1ccc(-c2ccccc2)cc1", is_purchasable=my_check)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from rdkit import Chem

from fragmentretro.constraints import ConstraintConfig
from fragmentretro.reaction_library import Reaction, ReactionLibrary
from fragmentretro.utils.logging_config import logger


@dataclass
class RetroSynthNode:
    """Node in a SMARTS-based retrosynthesis tree.

    Attributes:
        smiles: Canonical SMILES of this molecule.
        children: Child nodes (reactants from retrosynthetic disconnection).
        reaction: The Reaction applied to produce children.
        depth: Distance from root.
        is_building_block: True if this molecule is purchasable.
    """

    smiles: str
    children: list[RetroSynthNode] = field(default_factory=list)
    reaction: Optional[Reaction] = None
    depth: int = 0
    is_building_block: bool = False

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_solved(self) -> bool:
        """True if all leaves are building blocks."""
        if self.is_leaf:
            return self.is_building_block
        return all(c.is_solved for c in self.children)

    @property
    def num_steps(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + sum(c.num_steps for c in self.children)

    @property
    def longest_linear_sequence(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.longest_linear_sequence for c in self.children)

    @property
    def num_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.num_leaves for c in self.children)

    @property
    def avg_reliability(self) -> float:
        """Average reliability of all reactions in the tree."""
        scores: list[float] = []
        self._collect_reliabilities(scores)
        return sum(scores) / len(scores) if scores else 1.0

    def _collect_reliabilities(self, acc: list[float]) -> None:
        if self.reaction:
            acc.append(self.reaction.reliability)
        for c in self.children:
            c._collect_reliabilities(acc)

    def to_dict(self) -> dict:
        d: dict = {
            "smiles": self.smiles,
            "depth": self.depth,
            "is_building_block": self.is_building_block,
        }
        if self.reaction:
            d["reaction"] = {
                "id": self.reaction.id,
                "name": self.reaction.name,
                "class": self.reaction.reaction_class,
                "reliability": self.reaction.reliability,
            }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.depth == 0 and self.children:
            d["metrics"] = {
                "total_steps": self.num_steps,
                "longest_linear_sequence": self.longest_linear_sequence,
                "num_building_blocks": self.num_leaves,
                "avg_reliability": round(self.avg_reliability, 3),
                "solved": self.is_solved,
            }
        return d

    def pretty_print(self, indent: int = 0) -> str:
        prefix = "  " * indent
        tag = " [BB]" if self.is_building_block else ""
        lines = [f"{prefix}{'└─ ' if indent > 0 else ''}{self.smiles}{tag}"]
        if self.reaction:
            lines.append(f"{prefix}   ↑ {self.reaction.name} (r={self.reaction.reliability})")
        for child in self.children:
            lines.append(child.pretty_print(indent + 1))
        if indent == 0 and self.children:
            lines.append(f"\n--- Metrics ---")
            lines.append(f"Steps: {self.num_steps} | LLS: {self.longest_linear_sequence}")
            lines.append(f"BBs: {self.num_leaves} | Avg reliability: {self.avg_reliability:.2f}")
            lines.append(f"Solved: {self.is_solved}")
        return "\n".join(lines)


class SmartsRetrosynthesis:
    """Standalone retrosynthesis engine using SMARTS reaction patterns.

    Builds retrosynthesis trees by recursively applying reactions in reverse.
    Uses best-first expansion (highest reliability first) with global
    deduplication and early termination on solved routes.

    Args:
        library: ReactionLibrary to use for disconnections.
        max_depth: Maximum tree depth (default: 5).
        max_nodes: Maximum total nodes explored (default: 2000).
        constraints: Optional ConstraintConfig for filtering.
    """

    def __init__(
        self,
        library: ReactionLibrary,
        max_depth: int = 5,
        max_nodes: int = 2000,
        constraints: ConstraintConfig | None = None,
    ):
        self.library = library
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.constraints = constraints
        self._nodes_explored = 0

        # Apply constraints to library if provided
        if constraints:
            self.library = constraints.get_filtered_library(library)

        # Pre-sort reactions by reliability (descending) for best-first search
        self._sorted_reactions = sorted(
            self.library.reactions,
            key=lambda r: r.reliability,
            reverse=True,
        )

    def retrosynthesise(
        self,
        target_smiles: str,
        is_purchasable: Callable[[str], bool] | None = None,
        max_routes: int = 10,
    ) -> list[RetroSynthNode]:
        """Find retrosynthetic routes for a target molecule.

        Args:
            target_smiles: SMILES of the target molecule.
            is_purchasable: Callback that returns True if a SMILES is a purchasable BB.
                If None, leaves are never marked as BBs (tree is purely structural).
            max_routes: Maximum number of complete routes to return.

        Returns:
            List of RetroSynthNode trees, sorted by (solved, reliability).
        """
        self._nodes_explored = 0
        # Cache is_purchasable to avoid redundant BB checks
        self._purchasable_cache: dict[str, bool] = {}
        # Track globally expanded (smiles, depth) to avoid re-expansion
        self._expanded: set[str] = set()

        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            logger.warning(f"[SmartsRetro] Invalid SMILES: {target_smiles}")
            return []

        canonical = Chem.MolToSmiles(mol)

        # Check if target is itself a BB
        if is_purchasable and self._check_purchasable(canonical, is_purchasable):
            return [RetroSynthNode(smiles=canonical, is_building_block=True)]

        # Suppress RDKit warnings during search (reverse SMARTS produce
        # many expected valence / aromatic warnings that slow down via I/O)
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.*')

        try:
            routes = self._expand(
                canonical,
                depth=0,
                is_purchasable=is_purchasable,
                visited=frozenset(),
            )
        finally:
            rdBase.EnableLog('rdApp.*')

        # Sort: solved first, then by avg_reliability descending
        routes.sort(key=lambda r: (r.is_solved, r.avg_reliability), reverse=True)

        return routes[:max_routes]

    def _check_purchasable(
        self, smiles: str, is_purchasable: Callable[[str], bool]
    ) -> bool:
        """Cached purchasability check."""
        if smiles not in self._purchasable_cache:
            try:
                self._purchasable_cache[smiles] = is_purchasable(smiles)
            except Exception:
                self._purchasable_cache[smiles] = False
        return self._purchasable_cache[smiles]

    def _expand(
        self,
        smiles: str,
        depth: int,
        is_purchasable: Callable[[str], bool] | None,
        visited: frozenset[str],
    ) -> list[RetroSynthNode]:
        """Recursively expand a molecule into retrosynthesis trees."""
        self._nodes_explored += 1

        if self._nodes_explored > self.max_nodes:
            return [RetroSynthNode(smiles=smiles, depth=depth)]

        if depth >= self.max_depth:
            is_bb = (
                self._check_purchasable(smiles, is_purchasable)
                if is_purchasable
                else False
            )
            return [RetroSynthNode(smiles=smiles, depth=depth, is_building_block=is_bb)]

        if smiles in visited:
            # Cycle detection
            return [RetroSynthNode(smiles=smiles, depth=depth)]

        # Global deduplication: skip if already expanded at same or shallower depth
        dedup_key = smiles
        if dedup_key in self._expanded:
            is_bb = (
                self._check_purchasable(smiles, is_purchasable)
                if is_purchasable
                else False
            )
            return [RetroSynthNode(smiles=smiles, depth=depth, is_building_block=is_bb)]
        self._expanded.add(dedup_key)

        new_visited = visited | {smiles}

        # Pre-filter: only attempt reactions whose product SMARTS matches
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [RetroSynthNode(smiles=smiles, depth=depth)]

        routes: list[RetroSynthNode] = []
        found_solved = False

        for rxn in self._sorted_reactions:
            if self._nodes_explored > self.max_nodes:
                break

            # Early termination: once we have a solved route, skip lower-reliability reactions
            if found_solved:
                break

            if self.constraints and not self.constraints.is_reaction_allowed(rxn):
                continue

            # Cheap substructure pre-filter before expensive reverse application
            if not rxn.matches_product(mol):
                continue

            reactant_sets = rxn.apply_reverse(smiles)
            if not reactant_sets:
                continue

            for reactants in reactant_sets:
                if self._nodes_explored > self.max_nodes:
                    break

                # Check LLS constraint early
                if self.constraints and self.constraints.max_lls is not None:
                    if depth + 1 > self.constraints.max_lls:
                        continue

                # Recursively expand each reactant
                child_options: list[list[RetroSynthNode]] = []
                valid = True

                for r_smi in reactants:
                    # Canonicalize
                    r_mol = Chem.MolFromSmiles(r_smi)
                    if r_mol is None:
                        valid = False
                        break
                    r_canonical = Chem.MolToSmiles(r_mol)

                    # Skip fragments same as or larger than parent (spurious match)
                    if r_mol.GetNumHeavyAtoms() >= mol.GetNumHeavyAtoms():
                        valid = False
                        break

                    if is_purchasable and self._check_purchasable(
                        r_canonical, is_purchasable
                    ):
                        child_options.append([
                            RetroSynthNode(
                                smiles=r_canonical,
                                depth=depth + 1,
                                is_building_block=True,
                            )
                        ])
                    else:
                        sub_trees = self._expand(
                            r_canonical, depth + 1, is_purchasable, new_visited
                        )
                        if sub_trees:
                            child_options.append(sub_trees)
                        else:
                            valid = False
                            break

                if not valid or not child_options:
                    continue

                # Take best child from each position (greedy)
                best_children: list[RetroSynthNode] = []
                for opts in child_options:
                    best = max(opts, key=lambda n: (n.is_solved, n.avg_reliability))
                    best_children.append(best)

                node = RetroSynthNode(
                    smiles=smiles,
                    children=best_children,
                    reaction=rxn,
                    depth=depth,
                )

                # Check total steps constraint
                if self.constraints and self.constraints.max_steps is not None:
                    if node.num_steps > self.constraints.max_steps:
                        continue

                routes.append(node)

                if node.is_solved:
                    found_solved = True
                    break

        if not routes:
            is_bb = (
                self._check_purchasable(smiles, is_purchasable)
                if is_purchasable
                else False
            )
            return [RetroSynthNode(smiles=smiles, depth=depth, is_building_block=is_bb)]

        return routes
