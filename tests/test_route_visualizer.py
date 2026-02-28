"""Tests for route_visualizer."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from fragmentretro.fragmenter import BRICSFragmenter, rBRICSFragmenter
from fragmentretro.retro_dag import RetroNode, build_best_dag, build_dags
from fragmentretro.reaction_mapping import get_reaction_info
from fragmentretro.route_visualizer import draw_route, draw_all_routes


@pytest.fixture
def two_frag_dag():
    frag = BRICSFragmenter("CC(=O)Nc1ccccc1")
    solution = [(i,) for i in range(frag.num_fragments)]
    comb_bbs = {(i,): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}
    return build_best_dag(solution, frag, comb_bbs)


@pytest.fixture
def multi_frag_dag():
    frag = rBRICSFragmenter("CC(=O)Nc1ccc(-c2ccccc2)cc1")
    if frag.num_fragments < 3:
        pytest.skip("Need 3+ fragments")
    solution = [(i,) for i in range(frag.num_fragments)]
    comb_bbs = {(i,): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}
    return build_best_dag(solution, frag, comb_bbs)


@pytest.fixture
def leaf_dag():
    return RetroNode(smiles="CCN", fragments=((0,),), bb_smiles={"CCN"})


def test_draw_returns_image(two_frag_dag):
    img = draw_route(two_frag_dag)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0


def test_draw_leaf_node(leaf_dag):
    img = draw_route(leaf_dag, title="Single BB")
    assert isinstance(img, Image.Image)


def test_draw_saves_png(two_frag_dag):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_route.png"
        draw_route(two_frag_dag, output_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000  # non-trivial file


def test_draw_saves_svg(two_frag_dag):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_route.svg"
        draw_route(two_frag_dag, output_path=path)
        assert path.exists()


def test_draw_multi_fragment(multi_frag_dag):
    img = draw_route(multi_frag_dag)
    assert isinstance(img, Image.Image)
    # Should be wider for more fragments
    assert img.size[0] > 500


def test_draw_custom_figsize(two_frag_dag):
    img = draw_route(two_frag_dag, figsize=(4, 3), dpi=72)
    assert isinstance(img, Image.Image)


def test_draw_all_routes():
    frag = rBRICSFragmenter("CC(=O)Nc1ccc(OC)cc1")
    if frag.num_fragments < 3:
        pytest.skip("Need 3+ fragments")
    solution = [(i,) for i in range(frag.num_fragments)]
    comb_bbs = {(i,): {frag.fragment_graph.nodes[i]["smiles"]} for i in range(frag.num_fragments)}
    routes = build_dags(solution, frag, comb_bbs, max_routes=3)
    assert len(routes) >= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = draw_all_routes(routes, tmpdir, prefix="test")
        assert len(paths) == len(routes)
        for p in paths:
            assert p.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
