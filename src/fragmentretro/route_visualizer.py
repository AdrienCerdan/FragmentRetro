"""Visualize retrosynthetic routes as annotated tree diagrams.

Renders RetroNode DAGs as publication-quality images with:
  - 2D molecule structures at each node (RDKit)
  - Reaction name and BRICS bond type labels on edges
  - BB indicators on leaf nodes
  - Synthesis metrics summary

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

from fragmentretro.retro_dag import RetroNode
from fragmentretro.utils.logging_config import logger

# Use non-interactive backend when saving to file
matplotlib.use("Agg")


def _mol_to_array(smiles: str, size: tuple[int, int] = (250, 200)) -> np.ndarray:
    """Render a SMILES string to a numpy RGB array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback: draw a placeholder
        img = Image.new("RGB", size, (240, 240, 240))
    else:
        img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def _collect_nodes_bfs(root: RetroNode) -> list[RetroNode]:
    """BFS traversal returning all nodes."""
    queue = [root]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _assign_positions(
    node: RetroNode,
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

    # Recurse children, placing them left-to-right
    child_positions = []
    current_x = x
    for i, child in enumerate(node.children):
        positions, rightmost = _assign_positions(
            child, current_x, y - y_spacing, x_spacing, y_spacing, positions
        )
        child_positions.append(positions[id(child)])
        current_x = rightmost + x_spacing

    # Center parent above its children
    avg_x = sum(cp[0] for cp in child_positions) / len(child_positions)
    positions[id(node)] = (avg_x, y)
    return positions, current_x - x_spacing


def draw_route(
    root: RetroNode,
    output_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    mol_size: tuple[int, int] = (250, 200),
    dpi: int = 150,
    title: str | None = None,
) -> Image.Image:
    """Draw a retrosynthetic route as a tree diagram with molecule structures.

    Args:
        root: Root RetroNode of the route.
        output_path: If provided, save the figure to this path (PNG/PDF/SVG).
        figsize: Figure size in inches. Auto-calculated if None.
        mol_size: Pixel size for each molecule drawing.
        dpi: Resolution for output.
        title: Optional title. Defaults to "Retrosynthetic Route for {smiles}".

    Returns:
        PIL Image of the rendered route.
    """
    all_nodes = _collect_nodes_bfs(root)
    positions, _ = _assign_positions(root, x_spacing=1.2, y_spacing=1.8)

    # Calculate figure dimensions
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

    # Title
    if title is None:
        title = f"Retrosynthetic Route for {root.smiles}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Scale for molecule image placement
    img_w = 0.45  # width in data coordinates
    img_h = img_w * mol_size[1] / mol_size[0]

    # Draw edges first (behind nodes)
    for node in all_nodes:
        if node.children:
            px, py = positions[id(node)]
            for child in node.children:
                cx, cy = positions[id(child)]
                # Draw arrow (retrosynthetic direction: parent → children)
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

            # Reaction label at midpoint of parent
            if node.reaction_info and node.bond_type:
                label = f"{node.reaction_info.name}\n[{node.bond_type[0]}-{node.bond_type[1]}]"
                # Place label to the right of parent node
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

        # Extent: [left, right, bottom, top]
        extent = [x - img_w / 2, x + img_w / 2, y - img_h / 2, y + img_h / 2]
        ax.imshow(mol_img, extent=extent, aspect="auto", zorder=3)

        # Border around molecule
        border_color = "#4CAF50" if node.is_leaf else "#2196F3"
        border_width = 2.0 if node.is_leaf else 1.5
        rect = mpatches.FancyBboxPatch(
            (x - img_w / 2, y - img_h / 2), img_w, img_h,
            boxstyle="round,pad=0.02",
            linewidth=border_width,
            edgecolor=border_color,
            facecolor="none",
            zorder=4,
        )
        ax.add_patch(rect)

        # Label: SMILES below molecule (truncated)
        display_smi = node.smiles if len(node.smiles) <= 35 else node.smiles[:32] + "..."
        ax.text(
            x, y - img_h / 2 - 0.06,
            display_smi,
            fontsize=6, ha="center", va="top",
            color="#333333",
            fontfamily="monospace",
        )

        # BB badge on leaves
        if node.is_leaf and node.bb_smiles:
            n_bb = len(node.bb_smiles)
            badge_text = f"BB ({n_bb})" if n_bb <= 99 else "BB (99+)"
            ax.text(
                x + img_w / 2 - 0.02, y + img_h / 2 - 0.02,
                badge_text,
                fontsize=6, ha="right", va="top",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#4CAF50", alpha=0.9),
                zorder=5,
            )

    # Metrics box (bottom-right)
    if not root.is_leaf:
        metrics_text = (
            f"Total steps: {root.num_steps}\n"
            f"LLS: {root.longest_linear_sequence}\n"
            f"BBs: {root.num_leaves}\n"
            f"Convergence: {root.convergence_score:.2f}"
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
        mpatches.Patch(facecolor="none", edgecolor="#4CAF50", linewidth=2.0, label="Building Block"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    buf.close()

    # Optionally save
    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"[Visualizer] Route saved to {output_path}")

    plt.close(fig)
    return pil_img


def draw_all_routes(
    routes: list[RetroNode],
    output_dir: str | Path,
    prefix: str = "route",
    **kwargs,
) -> list[Path]:
    """Draw multiple routes, saving each as a separate image.

    Args:
        routes: List of RetroNode trees.
        output_dir: Directory to save images.
        prefix: Filename prefix.
        **kwargs: Passed to draw_route().

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, route in enumerate(routes):
        path = output_dir / f"{prefix}_{i:02d}.png"
        draw_route(route, output_path=path, **kwargs)
        paths.append(path)
    return paths
