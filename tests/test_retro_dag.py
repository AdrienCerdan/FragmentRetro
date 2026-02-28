"""Tests for retrosynthetic DAG construction."""

import pytest
import networkx as nx

from fragmentretro.fragmenter import rBRICSFragmenter, BRICSFragmenter
from fragmentretro.retro_dag import (
    RetroNode,
    build_dags,
    build_best_dag,
    dag_to_networkx,
    _build_group_graph,
)
from fragmentretro.reaction_mapping import get_reaction_info, ReactionInfo


# --- Fixtures ---

@pytest.fixture
def simple_fragmenter():
    """A molecule that fragments into 2-3 pieces with BRICS."""
    return BRICSFragmenter("CC(=O)Nc1ccccc1")  # acetanilide: amide bond


@pytest.fixture
def three_frag_fragmenter():
    """A molecule that fragments into 3+ pieces."""
    return rBRICSFragmenter("CC(=O)Nc1ccc(OC)cc1")  # p-methoxyacetanilide


# --- reaction_mapping tests ---

def test_known_reaction():
    info = get_reaction_info("1", "5")
    assert info.name == "Amide coupling"
    assert info.forward_class == "acylation"


def test_unknown_reaction():
    info = get_reaction_info("999", "888")
    assert info.name == "Unknown disconnection"


def test_symmetric_lookup():
    assert get_reaction_info("1", "5") == get_reaction_info("5", "1")


# --- RetroNode tests ---

def test_leaf_node():
    node = RetroNode(smiles="CCN", fragments=((0,),), bb_smiles={"CCN"})
    assert node.is_leaf is True
    assert node.num_steps == 0


def test_internal_node():
    child1 = RetroNode(smiles="CC", fragments=((0,),))
    child2 = RetroNode(smiles="N", fragments=((1,),))
    parent = RetroNode(
        smiles="CCN", fragments=((0,), (1,)),
        children=[child1, child2],
        bond_type=("4", "5"),
    )
    assert parent.is_leaf is False
    assert parent.num_steps == 1
    assert parent.longest_linear_sequence == 1
    assert parent.num_leaves == 2
    assert parent.convergence_score == 1.0  # 2 leaves, always 1.0


def test_to_dict_roundtrip():
    node = RetroNode(
        smiles="CCN", fragments=((0,), (1,)),
        children=[RetroNode(smiles="CC", fragments=((0,),), bb_smiles={"CC"})],
        bond_type=("4", "5"),
        reaction_info=get_reaction_info("4", "5"),
    )
    d = node.to_dict()
    assert d["smiles"] == "CCN"
    assert "children" in d
    assert d["children"][0]["bb_smiles"] == ["CC"]
    assert d["reaction"]["name"] == "Reductive amination"


def test_pretty_print():
    child = RetroNode(smiles="CC", fragments=((0,),), bb_smiles={"CC"}, depth=1)
    parent = RetroNode(
        smiles="CCN", fragments=((0,), (1,)),
        children=[child],
        bond_type=("1", "5"),
        reaction_info=get_reaction_info("1", "5"),
    )
    text = parent.pretty_print()
    assert "CCN" in text
    assert "Amide coupling" in text
    assert "CC" in text


# --- Group graph tests ---

def test_group_graph_structure(simple_fragmenter):
    frag = simple_fragmenter
    if frag.num_fragments < 2:
        pytest.skip("Not enough fragments for group graph test")
    # Create a trivial solution: each fragment is its own group
    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    G = _build_group_graph(solution, frag)
    assert G.number_of_nodes() == len(solution)
    # Should have at least 1 edge
    assert G.number_of_edges() >= 1
    # All edges should have bond_type
    for _, _, data in G.edges(data=True):
        assert "bond_type" in data


# --- DAG building tests ---

def test_build_dags_single_fragment():
    """Single-fragment solution = target is a BB."""
    frag = BRICSFragmenter("CCO")  # ethanol — may not fragment
    if frag.num_fragments == 1:
        solution = [(0,)]
        comb_bbs = {(0,): {"CCO"}}
        dags = build_dags(solution, frag, comb_bbs)
        assert len(dags) == 1
        assert dags[0].is_leaf
        assert dags[0].bb_smiles == {"CCO"}


def test_build_dags_two_fragments(simple_fragmenter):
    frag = simple_fragmenter
    if frag.num_fragments < 2:
        pytest.skip("Not enough fragments")

    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    comb_bbs = {cast_comb(i): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}

    dags = build_dags(solution, frag, comb_bbs)
    assert len(dags) >= 1

    root = dags[0]
    assert root.smiles == frag.original_smiles
    assert not root.is_leaf
    assert root.reaction_info is not None
    assert len(root.children) == 2


def test_build_best_dag(simple_fragmenter):
    frag = simple_fragmenter
    if frag.num_fragments < 2:
        pytest.skip("Not enough fragments")

    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    comb_bbs = {cast_comb(i): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}

    best = build_best_dag(solution, frag, comb_bbs)
    assert best.smiles == frag.original_smiles
    assert best.num_steps >= 1


def test_dag_to_networkx(simple_fragmenter):
    frag = simple_fragmenter
    if frag.num_fragments < 2:
        pytest.skip("Not enough fragments")

    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    comb_bbs = {cast_comb(i): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}

    best = build_best_dag(solution, frag, comb_bbs)
    G = dag_to_networkx(best)

    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == frag.num_fragments * 2 - 1 or G.number_of_nodes() >= frag.num_fragments
    # Root should have no predecessors
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    assert len(roots) == 1
    # Leaves should have no successors
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    assert len(leaves) >= 2


def test_three_fragment_dag(three_frag_fragmenter):
    """Test with 3+ fragments to verify recursive splitting."""
    frag = three_frag_fragmenter
    if frag.num_fragments < 3:
        pytest.skip("Need 3+ fragments")

    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    comb_bbs = {cast_comb(i): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}

    dags = build_dags(solution, frag, comb_bbs, max_routes=20)
    # With 3 fragments in a line, there should be exactly 2 orderings
    assert len(dags) >= 2

    for dag in dags:
        assert dag.smiles == frag.original_smiles
        assert dag.num_steps >= 2
        # All leaves should be single fragments
        _check_leaves(dag, frag.num_fragments)


def test_multiple_routes_differ(three_frag_fragmenter):
    """Different routes should have different first-cut bond types."""
    frag = three_frag_fragmenter
    if frag.num_fragments < 3:
        pytest.skip("Need 3+ fragments")

    solution = [cast_comb(i) for i in range(frag.num_fragments)]
    comb_bbs = {cast_comb(i): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}

    dags = build_dags(solution, frag, comb_bbs, max_routes=20)
    if len(dags) >= 2:
        # At least some routes should differ in structure
        structures = set()
        for dag in dags:
            structures.add(_route_signature(dag))
        assert len(structures) >= 2


def test_convergence_linear_vs_balanced():
    """A linear chain of 4 leaves should score lower than balanced."""
    # Linear: root -> (A, root2) -> (B, root3) -> (C, D)
    d = RetroNode(smiles="D", fragments=((3,),), depth=3)
    c = RetroNode(smiles="C", fragments=((2,),), depth=3)
    cd = RetroNode(smiles="CD", fragments=((2,), (3,)), children=[c, d], bond_type=("4", "5"), depth=2)
    b = RetroNode(smiles="B", fragments=((1,),), depth=2)
    bcd = RetroNode(smiles="BCD", fragments=((1,), (2,), (3,)), children=[b, cd], bond_type=("1", "5"), depth=1)
    a = RetroNode(smiles="A", fragments=((0,),), depth=1)
    linear_root = RetroNode(
        smiles="ABCD", fragments=((0,), (1,), (2,), (3,)),
        children=[a, bcd], bond_type=("3", "4"), depth=0,
    )

    # Balanced: root -> (AB, CD), AB -> (A, B), CD -> (C, D)
    a2 = RetroNode(smiles="A", fragments=((0,),), depth=2)
    b2 = RetroNode(smiles="B", fragments=((1,),), depth=2)
    ab = RetroNode(smiles="AB", fragments=((0,), (1,)), children=[a2, b2], bond_type=("3", "4"), depth=1)
    c2 = RetroNode(smiles="C", fragments=((2,),), depth=2)
    d2 = RetroNode(smiles="D", fragments=((3,),), depth=2)
    cd2 = RetroNode(smiles="CD", fragments=((2,), (3,)), children=[c2, d2], bond_type=("4", "5"), depth=1)
    balanced_root = RetroNode(
        smiles="ABCD", fragments=((0,), (1,), (2,), (3,)),
        children=[ab, cd2], bond_type=("1", "5"), depth=0,
    )

    assert linear_root.num_leaves == balanced_root.num_leaves == 4
    assert linear_root.longest_linear_sequence == 3  # linear chain
    assert balanced_root.longest_linear_sequence == 2  # balanced
    assert balanced_root.convergence_score > linear_root.convergence_score
    assert balanced_root.convergence_score == 1.0  # perfectly balanced
    assert linear_root.convergence_score == 0.0  # fully linear


def test_to_dict_includes_metrics():
    """Root-level to_dict should include synthesis metrics."""
    child1 = RetroNode(smiles="CC", fragments=((0,),), depth=1)
    child2 = RetroNode(smiles="N", fragments=((1,),), depth=1)
    root = RetroNode(
        smiles="CCN", fragments=((0,), (1,)),
        children=[child1, child2],
        bond_type=("4", "5"),
        reaction_info=get_reaction_info("4", "5"),
        depth=0,
    )
    d = root.to_dict()
    assert "metrics" in d
    assert d["metrics"]["total_steps"] == 1
    assert d["metrics"]["longest_linear_sequence"] == 1
    assert d["metrics"]["num_building_blocks"] == 2
    assert d["metrics"]["convergence_score"] == 1.0


# --- Helpers ---

def cast_comb(i: int) -> tuple:
    return (i,)


def _check_leaves(node: RetroNode, expected_leaf_count: int) -> None:
    """Verify leaf count equals number of fragments."""
    leaves = []
    _collect_leaves(node, leaves)
    assert len(leaves) == expected_leaf_count


def _collect_leaves(node: RetroNode, acc: list) -> None:
    if node.is_leaf:
        acc.append(node)
    for child in node.children:
        _collect_leaves(child, acc)


def _route_signature(node: RetroNode) -> str:
    """Create a hashable signature of the route structure."""
    if node.is_leaf:
        return node.smiles
    child_sigs = sorted(str(_route_signature(c)) for c in node.children)
    return f"{node.bond_type}|{'|'.join(child_sigs)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
