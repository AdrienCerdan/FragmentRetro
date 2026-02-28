"""Retrosynthetic DAG (Directed Acyclic Graph) construction from FragmentRetro solutions.

Converts a flat fragment-set solution into an ordered sequence of retrosynthetic
disconnections, producing a tree-structured DAG where:
  - Root = target molecule
  - Internal nodes = synthetic intermediates
  - Leaves = building blocks (BBs)
  - Edges = retrosynthetic steps labeled with reaction type

Algorithm:
  1. Build a "group graph" from the solution: nodes = fragment combos,
     edges = inter-group bonds from the fragment graph.
  2. Recursively split the group graph by removing one edge at a time.
     Each removal produces two connected components, each mapping to an
     intermediate SMILES via the fragmenter.
  3. Enumerate all valid orderings (routes) or return the best one
     ranked by bond feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import networkx as nx

from fragmentretro.bond_feasibility import get_bond_feasibility
from fragmentretro.fragmenter_base import Fragmenter
from fragmentretro.reaction_mapping import ReactionInfo, get_reaction_info
from fragmentretro.retrosynthesis import Retrosynthesis
from fragmentretro.typing import CombBBsDictType, CombType, SolutionType
from fragmentretro.utils.logging_config import logger


@dataclass
class RetroNode:
    """A node in the retrosynthetic DAG.

    Attributes:
        smiles: SMILES of this molecule/intermediate.
        fragments: The fragment combo indices merged at this node.
        children: Child nodes (products of retrosynthetic disconnection).
        reaction_info: Reaction that disconnects this node into children.
        bond_type: BRICS (env1, env2) label of the disconnected bond.
        depth: Distance from root (target = 0).
        bb_smiles: If leaf, set of matched building block SMILES.
    """

    smiles: str
    fragments: tuple[CombType, ...]
    children: list[RetroNode] = field(default_factory=list)
    reaction_info: ReactionInfo | None = None
    bond_type: tuple[str, str] | None = None
    depth: int = 0
    bb_smiles: set[str] | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def num_steps(self) -> int:
        """Total number of synthetic steps (edges) in this subtree."""
        if self.is_leaf:
            return 0
        return 1 + sum(c.num_steps for c in self.children)

    @property
    def longest_linear_sequence(self) -> int:
        """Longest linear sequence (LLS): max depth from root to any leaf.

        This is the critical path — the minimum number of sequential steps
        needed even with unlimited parallel capacity.
        """
        if self.is_leaf:
            return 0
        return 1 + max(c.longest_linear_sequence for c in self.children)

    @property
    def num_leaves(self) -> int:
        """Number of leaf nodes (building blocks)."""
        if self.is_leaf:
            return 1
        return sum(c.num_leaves for c in self.children)

    @property
    def convergence_score(self) -> float:
        """How convergent the synthesis is, in [0, 1].

        1.0 = maximally convergent (balanced binary tree, LLS = log2(leaves))
        0.0 = fully linear (chain, LLS = leaves - 1)

        For a single BB (no steps), returns 1.0.
        For 2 BBs (1 step), always returns 1.0 (no choice to make).
        """
        n_leaves = self.num_leaves
        if n_leaves <= 2:
            return 1.0
        lls = self.longest_linear_sequence
        # Worst case (linear): LLS = n_leaves - 1
        # Best case (balanced): LLS = ceil(log2(n_leaves))
        import math
        best_lls = math.ceil(math.log2(n_leaves))
        worst_lls = n_leaves - 1
        if worst_lls == best_lls:
            return 1.0
        return (worst_lls - lls) / (worst_lls - best_lls)

    def to_dict(self) -> dict:
        """Serialize to a nested dictionary."""
        d: dict = {
            "smiles": self.smiles,
            "fragments": [list(c) for c in self.fragments],
            "depth": self.depth,
        }
        if self.bond_type:
            d["bond_type"] = self.bond_type
        if self.reaction_info:
            d["reaction"] = {
                "name": self.reaction_info.name,
                "description": self.reaction_info.description,
                "forward_class": self.reaction_info.forward_class,
            }
        if self.bb_smiles:
            d["bb_smiles"] = sorted(self.bb_smiles)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        # Include synthesis metrics at root
        if self.depth == 0:
            d["metrics"] = {
                "total_steps": self.num_steps,
                "longest_linear_sequence": self.longest_linear_sequence,
                "num_building_blocks": self.num_leaves,
                "convergence_score": round(self.convergence_score, 3),
            }
        return d

    def pretty_print(self, indent: int = 0) -> str:
        """Human-readable tree representation."""
        prefix = "  " * indent
        lines = [f"{prefix}{'└─ ' if indent > 0 else ''}{self.smiles}"]
        if self.reaction_info and self.bond_type:
            lines.append(
                f"{prefix}   ↑ {self.reaction_info.name} "
                f"[BRICS {self.bond_type[0]}-{self.bond_type[1]}]"
            )
        if self.bb_smiles:
            bb_list = ", ".join(sorted(self.bb_smiles)[:3])
            suffix = "..." if self.bb_smiles and len(self.bb_smiles) > 3 else ""
            lines.append(f"{prefix}   BBs: {bb_list}{suffix}")
        for child in self.children:
            lines.append(child.pretty_print(indent + 1))
        # Show metrics at root
        if indent == 0 and not self.is_leaf:
            lines.append(f"\n--- Synthesis metrics ---")
            lines.append(f"Total steps: {self.num_steps}")
            lines.append(f"Longest linear sequence (LLS): {self.longest_linear_sequence}")
            lines.append(f"Building blocks: {self.num_leaves}")
            lines.append(f"Convergence: {self.convergence_score:.2f} (1.0=fully convergent, 0.0=fully linear)")
        return "\n".join(lines)


def _build_group_graph(
    solution: SolutionType, fragmenter: Fragmenter
) -> nx.Graph:
    """Build a graph where nodes are solution combs and edges are inter-group fragment bonds.

    Each edge carries the BRICS bond_type and the fragment-graph edge data.
    """
    G = nx.Graph()
    for i, comb in enumerate(solution):
        G.add_node(i, comb=comb)

    for i, comb_i in enumerate(solution):
        for j, comb_j in enumerate(solution):
            if j <= i:
                continue
            for node_a in comb_i:
                for node_b in comb_j:
                    edge_data = fragmenter.fragment_graph.get_edge_data(node_a, node_b)
                    if edge_data and not G.has_edge(i, j):
                        G.add_edge(
                            i, j,
                            bond_type=edge_data["bond_type"],
                            frag_nodes=(node_a, node_b),
                        )
    return G


def _merge_combs(combs: list[CombType]) -> CombType:
    """Merge multiple CombTypes into a single sorted tuple."""
    merged: list[int] = []
    for c in combs:
        merged.extend(c)
    return cast(CombType, tuple(sorted(merged)))


def _get_smiles_for_combs(
    combs: list[CombType], fragmenter: Fragmenter
) -> str:
    """Get the SMILES for a merged set of fragment combinations."""
    merged = _merge_combs(combs)
    return fragmenter.get_combination_smiles(merged)


def _build_dag_recursive(
    group_indices: list[int],
    solution: SolutionType,
    group_graph: nx.Graph,
    fragmenter: Fragmenter,
    comb_bbs_dict: CombBBsDictType,
    depth: int = 0,
) -> list[RetroNode]:
    """Recursively build all possible DAGs by trying each edge cut.

    Returns a list of RetroNode trees, one per valid disconnection ordering.
    """
    combs = [solution[i] for i in group_indices]
    merged_smiles = _get_smiles_for_combs(combs, fragmenter)

    # Base case: single group = leaf (BB)
    if len(group_indices) == 1:
        comb = solution[group_indices[0]]
        bb = comb_bbs_dict.get(comb)
        return [RetroNode(
            smiles=merged_smiles,
            fragments=tuple(combs),
            depth=depth,
            bb_smiles=bb,
        )]

    # Get subgraph for current groups
    sub = group_graph.subgraph(group_indices)
    edges = list(sub.edges(data=True))

    if not edges:
        # Disconnected groups — shouldn't happen for valid solutions
        logger.warning(f"[RetroDAG] No edges between groups {group_indices}")
        return [RetroNode(smiles=merged_smiles, fragments=tuple(combs), depth=depth)]

    routes: list[RetroNode] = []

    for u, v, edata in edges:
        # Check that removing this edge still leaves both parts connected
        # (i.e., the edge is a bridge OR we only cut bridges)
        sub_copy = sub.copy()
        sub_copy.remove_edge(u, v)
        components = list(nx.connected_components(sub_copy))

        if len(components) != 2:
            # Removing this edge doesn't cleanly split into 2 parts
            # (multi-edge between same groups, or creates 3+ components)
            # Still valid if it creates exactly 2 components
            if len(components) < 2:
                continue
            # For >2 components, skip this cut (would need multi-way split)
            continue

        left_indices = sorted(components[0])
        right_indices = sorted(components[1])

        bond_type = edata["bond_type"]
        reaction_info = get_reaction_info(bond_type[0], bond_type[1])

        # Recurse on both sides
        left_trees = _build_dag_recursive(
            left_indices, solution, group_graph, fragmenter, comb_bbs_dict, depth + 1
        )
        right_trees = _build_dag_recursive(
            right_indices, solution, group_graph, fragmenter, comb_bbs_dict, depth + 1
        )

        # Combine all left × right possibilities
        for left_node in left_trees:
            for right_node in right_trees:
                root = RetroNode(
                    smiles=merged_smiles,
                    fragments=tuple(combs),
                    children=[left_node, right_node],
                    reaction_info=reaction_info,
                    bond_type=bond_type,
                    depth=depth,
                )
                routes.append(root)

    return routes if routes else [RetroNode(
        smiles=merged_smiles, fragments=tuple(combs), depth=depth
    )]


def build_dags(
    solution: SolutionType,
    fragmenter: Fragmenter,
    comb_bbs_dict: CombBBsDictType,
    max_routes: int = 10,
) -> list[RetroNode]:
    """Build all retrosynthetic DAGs for a given solution.

    Args:
        solution: A retrosynthesis solution (list of CombTypes).
        fragmenter: The Fragmenter that produced the fragment graph.
        comb_bbs_dict: Mapping from CombType to matched BB SMILES.
        max_routes: Maximum number of routes to return.

    Returns:
        List of RetroNode trees, each representing a different disconnection ordering.
    """
    if len(solution) == 1:
        # Target is itself a BB
        comb = solution[0]
        smiles = fragmenter.get_combination_smiles(comb)
        return [RetroNode(
            smiles=smiles,
            fragments=(comb,),
            bb_smiles=comb_bbs_dict.get(comb),
        )]

    group_graph = _build_group_graph(solution, fragmenter)
    all_indices = list(range(len(solution)))

    routes = _build_dag_recursive(
        all_indices, solution, group_graph, fragmenter, comb_bbs_dict
    )

    return routes[:max_routes]


def build_best_dag(
    solution: SolutionType,
    fragmenter: Fragmenter,
    comb_bbs_dict: CombBBsDictType,
) -> RetroNode:
    """Build the single best DAG ranked by bond feasibility and convergence.

    Prefers routes that are more convergent (lower LLS) and cut
    the most feasible (easiest) bonds.

    Args:
        solution: A retrosynthesis solution.
        fragmenter: The Fragmenter instance.
        comb_bbs_dict: Mapping from CombType to matched BB SMILES.

    Returns:
        The highest-scoring RetroNode tree.
    """
    routes = build_dags(solution, fragmenter, comb_bbs_dict, max_routes=50)

    if not routes:
        raise ValueError("No valid DAGs could be built for this solution")

    def _score_route(node: RetroNode) -> float:
        """Score combining bond feasibility (60%) and convergence (40%)."""
        # Bond feasibility component
        bond_scores: list[float] = []
        _collect_bond_scores(node, bond_scores)
        avg_bond = sum(bond_scores) / len(bond_scores) if bond_scores else 1.0
        # Convergence component
        conv = node.convergence_score
        return 0.6 * avg_bond + 0.4 * conv

    return max(routes, key=_score_route)


def _collect_bond_scores(node: RetroNode, acc: list[float]) -> None:
    """Recursively collect bond feasibility scores from all cuts."""
    if node.bond_type:
        acc.append(get_bond_feasibility(node.bond_type[0], node.bond_type[1]))
    for child in node.children:
        _collect_bond_scores(child, acc)


def build_dag_from_retrosynthesis(
    retro: Retrosynthesis,
    solution: SolutionType,
) -> RetroNode:
    """Convenience: build the best DAG directly from a Retrosynthesis object.

    Args:
        retro: Completed Retrosynthesis object (after fragment_retrosynthesis()).
        solution: One solution from RetrosynthesisSolution.solutions.

    Returns:
        RetroNode tree for the best route.
    """
    return build_best_dag(solution, retro.fragmenter, retro.comb_bbs_dict)


def dag_to_networkx(root: RetroNode) -> nx.DiGraph:
    """Convert a RetroNode tree to a NetworkX DiGraph for visualization.

    Nodes have 'smiles', 'depth', 'bb_smiles' attributes.
    Edges have 'reaction_name', 'bond_type' attributes.
    """
    G = nx.DiGraph()
    node_id = 0

    def _add(node: RetroNode, parent_id: int | None) -> int:
        nonlocal node_id
        nid = node_id
        node_id += 1

        attrs = {"smiles": node.smiles, "depth": node.depth}
        if node.bb_smiles:
            attrs["bb_smiles"] = node.bb_smiles
        G.add_node(nid, **attrs)

        if parent_id is not None:
            edge_attrs: dict = {}
            if node.bond_type:
                edge_attrs["bond_type"] = node.bond_type
            if node.reaction_info:
                edge_attrs["reaction_name"] = node.reaction_info.name
            # Edge direction: parent → child (retrosynthetic direction)
            G.add_edge(parent_id, nid, **edge_attrs)

        for child in node.children:
            _add(child, nid)

        return nid

    _add(root, None)
    return G
