"""Visualize SMARTS-based retrosynthetic routes as annotated tree diagrams.

Renders RetroSynthNode trees (T3 engine) as publication-quality images with:
  - 2D molecule structures at each node (RDKit)
  - Reaction name, class, and reliability labels on edges
  - Building block indicators on leaf nodes
  - Synthesis metrics summary

Adapted from route_visualizer.py (which handles RetroNode DAGs from BRICS).

Requires: rdkit, matplotlib, Pillow (all in project deps).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from fragmentretro.smarts_retro import RetroSynthNode
from fragmentretro.utils.logging_config import logger

# Use non-interactive backend when saving to file
matplotlib.use("Agg")


def _mol_to_array(smiles: str, size: tuple[int, int] = (250, 200)) -> np.ndarray:
    """Render a SMILES string to a numpy RGB array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        img = Image.new("RGB", size, (240, 240, 240))
    else:
        img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def _collect_nodes_bfs(root: RetroSynthNode) -> list[RetroSynthNode]:
    """BFS traversal returning all nodes."""
    queue = [root]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _assign_positions(
    node: RetroSynthNode,
    x: float = 0.0,
    y: float = 0.0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.5,
    positions: dict | None = None,
) -> tuple[dict, float]:
    """Assign (x, y) positions to each node in a top-down tree layout.

    Returns dict mapping id(node) -> (x, y) and the rightmost x used.
    """
    if positions is None:
        positions = {}

    if node.is_leaf:
        positions[id(node)] = (x, y)
        return positions, x

    child_positions = []
    current_x = x
    for i, child in enumerate(node.children):
        positions, rightmost = _assign_positions(
            child, current_x, y - y_spacing, x_spacing, y_spacing, positions
        )
        child_positions.append(positions[id(child)])
        current_x = rightmost + x_spacing

    avg_x = sum(cp[0] for cp in child_positions) / len(child_positions)
    positions[id(node)] = (avg_x, y)
    return positions, current_x - x_spacing


def draw_t3_route(
    root: RetroSynthNode,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    mol_size: tuple[int, int] = (250, 200),
    dpi: int = 150,
    title: str | None = None,
) -> Image.Image:
    """Draw a SMARTS retrosynthetic route as a tree diagram.

    Args:
        root: Root RetroSynthNode.
        output_path: If provided, save the figure to this path (PNG/PDF/SVG).
        figsize: Figure size in inches. Auto-calculated if None.
        mol_size: Pixel size for each molecule drawing.
        dpi: Resolution for output.
        title: Optional title.

    Returns:
        PIL Image of the rendered route.
    """
    all_nodes = _collect_nodes_bfs(root)
    positions, _ = _assign_positions(root, x_spacing=1.2, y_spacing=1.8)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x_range = max(xs) - min(xs) + 2
    y_range = max(ys) - min(ys) + 2

    if figsize is None:
        figsize = (max(8, x_range * 3.5), max(6, y_range * 3.0))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    if title is None:
        status = "✓ Solved" if root.is_solved else "✗ Unsolved"
        title = f"T3 Retrosynthesis: {root.smiles[:60]}  [{status}]"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    img_w = 0.45
    img_h = img_w * mol_size[1] / mol_size[0]

    # Draw edges
    for node in all_nodes:
        if node.children:
            px, py = positions[id(node)]
            for child in node.children:
                cx, cy = positions[id(child)]
                ax.annotate(
                    "",
                    xy=(cx, cy + img_h / 2 + 0.05),
                    xytext=(px, py - img_h / 2 - 0.05),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#555555",
                        lw=1.8,
                        connectionstyle="arc3,rad=0.0",
                    ),
                )

            # Reaction label
            if node.reaction:
                rxn = node.reaction
                label = f"{rxn.name}\n[{rxn.reaction_class}] r={rxn.reliability:.2f}"
                ax.text(
                    px + img_w / 2 + 0.08, py - 0.15,
                    label,
                    fontsize=7,
                    color="#B03030",
                    ha="left", va="top",
                    fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF8F0",
                              edgecolor="#E0C0A0", alpha=0.9),
                )

    # Draw molecule images
    for node in all_nodes:
        x, y = positions[id(node)]
        mol_img = _mol_to_array(node.smiles, size=mol_size)

        extent = [x - img_w / 2, x + img_w / 2, y - img_h / 2, y + img_h / 2]
        ax.imshow(mol_img, extent=extent, aspect="auto", zorder=3)

        # Border
        if node.is_building_block:
            border_color = "#4CAF50"
            border_width = 2.5
        elif node.is_leaf:
            border_color = "#FF9800"  # Orange for unsolved leaves
            border_width = 2.0
        else:
            border_color = "#2196F3"
            border_width = 1.5

        rect = mpatches.FancyBboxPatch(
            (x - img_w / 2, y - img_h / 2), img_w, img_h,
            boxstyle="round,pad=0.02",
            linewidth=border_width,
            edgecolor=border_color,
            facecolor="none",
            zorder=4,
        )
        ax.add_patch(rect)

        # SMILES label
        display_smi = node.smiles if len(node.smiles) <= 35 else node.smiles[:32] + "..."
        ax.text(
            x, y - img_h / 2 - 0.06,
            display_smi,
            fontsize=6, ha="center", va="top",
            color="#333333",
            fontfamily="monospace",
        )

        # BB badge
        if node.is_building_block:
            ax.text(
                x + img_w / 2 - 0.02, y + img_h / 2 - 0.02,
                "BB ✓",
                fontsize=7, ha="right", va="top",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#4CAF50", alpha=0.9),
                zorder=5,
            )
        elif node.is_leaf and not node.is_building_block:
            ax.text(
                x + img_w / 2 - 0.02, y + img_h / 2 - 0.02,
                "Not in stock",
                fontsize=6, ha="right", va="top",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#FF5722", alpha=0.9),
                zorder=5,
            )

    # Metrics box
    if not root.is_leaf:
        metrics_text = (
            f"Solved: {'Yes' if root.is_solved else 'No'}\n"
            f"Steps: {root.num_steps}\n"
            f"LLS: {root.longest_linear_sequence}\n"
            f"BBs: {root.num_leaves}\n"
            f"Avg reliability: {root.avg_reliability:.3f}"
        )
        ax.text(
            0.98, 0.02, metrics_text,
            transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4FF",
                      edgecolor="#B0C4DE", alpha=0.95),
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="#2196F3", linewidth=1.5, label="Intermediate"),
        mpatches.Patch(facecolor="none", edgecolor="#4CAF50", linewidth=2.5, label="Building Block"),
        mpatches.Patch(facecolor="none", edgecolor="#FF9800", linewidth=2.0, label="Unsolved Leaf"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"[T3 Visualizer] Route saved to {output_path}")

    plt.close(fig)
    return pil_img


def draw_all_t3_routes(
    routes: list[RetroSynthNode],
    output_dir: str | Path,
    prefix: str = "route",
    **kwargs,
) -> list[Path]:
    """Draw multiple T3 routes, saving each as a separate image.

    Args:
        routes: List of RetroSynthNode trees.
        output_dir: Directory to save images.
        prefix: Filename prefix.
        **kwargs: Passed to draw_t3_route().

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, route in enumerate(routes):
        path = output_dir / f"{prefix}_{i:02d}.png"
        draw_t3_route(route, output_path=path, **kwargs)
        paths.append(path)
    return paths
