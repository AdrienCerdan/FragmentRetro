# FragmentRetro — Scoring, DAG Routes & Visualization

## Overview

This document covers extensions to FragmentRetro that transform it from a fragment-set retrosynthesis tool into a complete scoring and route-planning engine suitable for reinforcement learning (RL) integration and medicinal chemistry decision-making.

The original FragmentRetro algorithm (Shee & Morgunov, *J. Chem. Theory Comput.* 2026, 22, 972–980) uses BRICS/r-BRICS fragmentation to determine whether a target molecule can be reconstructed from commercially available building blocks (BBs). It achieves O(h²) complexity — far faster than tree-search methods (O(b^h)) — but only outputs *flat fragment sets*, not reaction sequences.

The extensions described here add:

1. **Three-tier scoring** — binary, continuous, and DAG-based — with increasing cost and information.
2. **Retrosynthetic DAGs** — ordered disconnection sequences with named reactions.
3. **Convergence analysis** — quantifying whether a route is convergent or linear.
4. **Route visualization** — publication-quality tree diagrams with 2D molecule structures.
5. **Batch scoring** — parallelized evaluation for Reinvent4 RL integration.

---

## Scientific Background

### Fragmentation-Based Retrosynthesis

Traditional retrosynthesis planners explore a tree of disconnections recursively: given target T, propose all possible single-step retrosynthetic transforms, evaluate each set of precursors, and recurse. This is O(b^h) where b is the branching factor and h is the tree depth.

FragmentRetro inverts this: it **fragments the target first** using BRICS/r-BRICS rules, then checks whether each fragment (or combination of fragments) exists as a substructure of any building block in the stock. The key insight is that fragment combinations at stage k can be pruned using results from stage k−1, yielding O(h²) complexity where h is the number of fragments.

**BRICS** (Breaking of Retrosynthetically Interesting Chemical Substructures; Degen et al., *ChemMedChem* 2008) defines 16 bond environments corresponding to known reaction types. **r-BRICS** extends this with additional rules for aliphatic chain cleavage, achieving 78.4% solved rate on USPTO-190 versus 53.2% for BRICS.

### From Fragment Sets to Reaction Routes

The original algorithm outputs *solutions* — partitions of the fragment graph into combinations that are each matched to a BB. A solution like `[(0,), (1,2), (3,)]` says: "fragment 0 maps to BB-A, fragments 1+2 together map to BB-B, fragment 3 maps to BB-C." But it does not specify in what *order* to assemble them.

The DAG extension solves this. Each edge in the fragment graph carries a BRICS bond-type label (e.g., L1–L5 = amide bond). These labels map directly to named reaction classes:

| BRICS Bond | Reaction | Reliability |
|------------|----------|-------------|
| L1–L5 | Amide coupling | Very high |
| L1–L3 | Ester formation | Very high |
| L5–L12 | Sulfonamide formation | Very high |
| L5–L14/L16 | Buchwald–Hartwig amination | High |
| L14–L16 | Suzuki coupling | High |
| L3–L4 | Williamson ether synthesis | High |
| L4–L5 | Reductive amination | High |
| L7a–L7b | Olefination (Wittig/HWE) | Moderate |
| L18–L19 | Ring fusion / annulation | Low |

By recursively splitting the "group graph" (where nodes = fragment combinations and edges = inter-group BRICS bonds), we enumerate all valid disconnection orderings. Each ordering is a binary tree = a retrosynthetic DAG.

### Convergent vs. Linear Synthesis

A critical concern for practical synthesis is whether the route is **convergent** (multiple branches can be executed in parallel) or **linear** (each step depends on the previous one).

For a route with N building blocks:

- **Fully linear**: LLS = N − 1 (worst case). Every intermediate must be made sequentially. Cumulative yield drops exponentially.
- **Fully convergent**: LLS = ⌈log₂(N)⌉ (best case). Balanced binary tree. Multiple branches can be run in parallel.

We define the **convergence score** as:

```
convergence = (worst_LLS - actual_LLS) / (worst_LLS - best_LLS)
```

where `worst_LLS = N - 1` and `best_LLS = ceil(log2(N))`.

A convergence of 1.0 means the route is maximally convergent (balanced tree); 0.0 means fully linear.

---

## Scoring API

Three scoring functions of increasing cost and detail:

### 1. Binary Scoring: `is_feasible()`

```python
from fragmentretro.scoring import is_feasible, load_compound_filter

cf = load_compound_filter("bb_properties.json")
result = is_feasible("CC(=O)Nc1ccccc1", cf)  # True/False
```

**Cost**: Lowest. Uses `binary_mode=True` (returns on first BB match per fragment) and `solution_cap=1`.

**Use case**: Fast filtering in RL reward computation. Typically 5–10 ms per molecule.

### 2. Continuous Scoring: `compute_score()`

```python
from fragmentretro.scoring import compute_score

score = compute_score("CC(=O)Nc1ccccc1", cf)  # float in [0, 1]
```

**Components** (weights tunable):

| Component | Weight | Description |
|-----------|--------|-------------|
| `step_score` | 0.35 | Fewer fragments in best solution → fewer steps → higher |
| `availability_score` | 0.30 | More BB matches per fragment → more sourcing options → higher |
| `bond_feasibility_score` | 0.35 | BRICS bond types mapped to reaction reliability → higher |

**Cost**: Medium. Full substructure matching but no DAG construction. Typically 5–20 ms.

**Use case**: Default RL reward signal. Good balance of speed and informativeness.

### 3. DAG Scoring: `compute_score_with_dag()`

```python
from fragmentretro.scoring import compute_score_with_dag, DAGScoreResult

result: DAGScoreResult = compute_score_with_dag("CC(=O)Nc1ccccc1", cf)
print(result.score)                  # 0.705
print(result.convergence_score)      # 1.0
print(result.longest_linear_sequence)  # 1
print(result.dag.pretty_print())     # full route tree
```

**Additional components** beyond `compute_score`:

| Component | Weight | Description |
|-----------|--------|-------------|
| `convergence_score` | 0.15 | Convergent routes score higher |
| `lls_score` | 0.15 | Shorter LLS relative to worst case → higher |

The other three components have reduced weights (0.25, 0.20, 0.25) to accommodate.

**`DAGScoreResult` fields**:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall score in [0, 1] |
| `feasible` | `bool` | Whether any solution was found |
| `total_steps` | `int` | Number of reactions in best route |
| `longest_linear_sequence` | `int` | Critical path length (LLS) |
| `num_building_blocks` | `int` | Number of BBs (leaves) |
| `convergence_score` | `float` | 1.0 = balanced, 0.0 = linear |
| `dag` | `RetroNode` | Full route tree (None if infeasible) |

**Cost**: Highest. Builds and ranks all valid DAGs. Typically 10–50 ms. For RL batches, consider using `compute_score` by default and reserving DAG scoring for top candidates.

---

## Retrosynthetic DAG Construction

### Building the DAG

```python
from fragmentretro.fragmenter import rBRICSFragmenter
from fragmentretro.retrosynthesis import Retrosynthesis
from fragmentretro.solutions import RetrosynthesisSolution
from fragmentretro.retro_dag import build_best_dag, build_dags, dag_to_networkx

# Standard FragmentRetro workflow
fragmenter = rBRICSFragmenter("CC(=O)Nc1ccc(-c2ccccc2)cc1")
retro = Retrosynthesis(fragmenter, mol_properties_path=props_path)
retro.fragment_retrosynthesis()
retro_sol = RetrosynthesisSolution(retro)
retro_sol.fill_solutions(solution_cap=5)

# Build best DAG for first solution
best_dag = build_best_dag(
    retro_sol.solutions[0], retro.fragmenter, retro.comb_bbs_dict
)

# Or enumerate all valid routes
all_routes = build_dags(
    retro_sol.solutions[0], retro.fragmenter, retro.comb_bbs_dict, max_routes=10
)
```

### RetroNode Properties

Each node in the DAG tree is a `RetroNode` with:

| Property | Description |
|----------|-------------|
| `.smiles` | SMILES of this intermediate |
| `.is_leaf` | True if building block |
| `.bb_smiles` | Set of matched BB SMILES (leaves only) |
| `.reaction_info` | `ReactionInfo(name, description, forward_class)` |
| `.bond_type` | BRICS labels `(env1, env2)` of the disconnected bond |
| `.num_steps` | Total reactions in subtree |
| `.longest_linear_sequence` | Max depth to any leaf |
| `.num_leaves` | Number of BBs |
| `.convergence_score` | Convergence metric [0, 1] |

### Serialization

```python
# To dictionary (JSON-serializable)
d = best_dag.to_dict()

# To NetworkX DiGraph
G = dag_to_networkx(best_dag)

# Human-readable text
print(best_dag.pretty_print())
```

---

## Route Visualization

### Drawing a Single Route

```python
from fragmentretro.route_visualizer import draw_route

# Returns PIL Image; optionally saves to file
img = draw_route(best_dag, output_path="route.png")

# Customize
img = draw_route(
    best_dag,
    output_path="route.svg",     # PNG, SVG, or PDF
    figsize=(12, 8),             # figure size in inches
    mol_size=(300, 250),         # molecule image resolution
    dpi=200,                     # output DPI
    title="My Retrosynthesis",
)
```

The rendered image includes:

- **2D molecule structures** (RDKit) at each node with color-coded borders (blue = intermediate, green = BB)
- **Reaction labels** with BRICS bond types on edges
- **BB badges** showing number of matched building blocks on leaf nodes
- **Synthesis metrics** panel (total steps, LLS, BBs, convergence)
- **Legend** distinguishing intermediates from building blocks

### Drawing Multiple Routes

```python
from fragmentretro.route_visualizer import draw_all_routes

paths = draw_all_routes(
    all_routes,               # list of RetroNode trees
    output_dir="routes/",     # directory for images
    prefix="aspirin",         # filename prefix
)
# Saves aspirin_00.png, aspirin_01.png, ...
```

---

## Batch Scoring for Reinvent4

Three batch functions matching the three scoring tiers:

```python
from fragmentretro.batch import (
    score_batch_binary,
    score_batch_continuous,
    score_batch_with_dag,
)

smiles_list = ["CCN", "CC(=O)Nc1ccccc1", "C1CC2CCCC3CCCC1C23"]

# Binary: list[float] of 0.0 or 1.0
scores = score_batch_binary(smiles_list, "bb_properties.json", max_workers=4)

# Continuous: list[float] in [0, 1]
scores = score_batch_continuous(smiles_list, "bb_properties.json", max_workers=4)

# DAG: list[dict] with score + route metrics
results = score_batch_with_dag(smiles_list, "bb_properties.json", max_workers=4)
for r in results:
    print(r["score"], r["convergence_score"], r["longest_linear_sequence"])
```

All batch functions use `ProcessPoolExecutor` with per-molecule timeouts (default: 10s binary, 30s continuous, 60s DAG). Molecules that timeout or error score 0.0.

**Reinvent4 integration pattern**:

```python
# In your Reinvent4 scoring component:
class FragmentRetroScore:
    def __init__(self, bb_props_path, mode="continuous"):
        self.bb_props_path = bb_props_path
        self.mode = mode

    def __call__(self, smiles_list):
        if self.mode == "binary":
            return score_batch_binary(smiles_list, self.bb_props_path)
        elif self.mode == "continuous":
            return score_batch_continuous(smiles_list, self.bb_props_path)
        elif self.mode == "dag":
            results = score_batch_with_dag(smiles_list, self.bb_props_path)
            return [r["score"] for r in results]
```

---

## Performance Optimizations

Several changes make FragmentRetro fast enough for RL batch scoring:

| Optimization | Location | Impact |
|-------------|----------|--------|
| `@lru_cache` on `convert_to_smarts()` and `addH_to_wildcard_neighbors()` | `substructure_matcher.py` | 2–5× speedup on repeated fragments |
| `binary_mode` early termination | `substructure_matcher.py` | Major speedup when only yes/no needed |
| Shared `CompoundFilter` singleton | `retrosynthesis.py` | Avoids JSON reload per molecule |
| `solution_cap` parameter | `solutions.py` | Skips combinatorial explosion |
| Process-level parallelism | `batch.py` | Scales with CPU cores |

---

## Bond Feasibility

The `bond_feasibility.py` module maps BRICS bond-type pairs to reaction reliability scores (0.0–1.0), addressing a limitation of the original algorithm: it verifies BB availability but not whether the reconnection chemistry is practical.

Scores reflect common synthetic practice:

- **1.0**: Routine reactions — amide coupling, ester formation, sulfonamide formation.
- **0.8**: Reliable with optimization — Buchwald–Hartwig, reductive amination.
- **0.6**: Standard cross-coupling — Suzuki, Negishi (require catalyst optimization).
- **0.3–0.4**: Challenging — ring fusions, unusual C–C bonds.

The `reaction_mapping.py` module provides human-readable names for the same bond types (e.g., L1–L5 → "Amide coupling", L14–L16 → "Suzuki coupling").

---

## Quick Start

```bash
# Install
git clone https://github.com/AdrienCerdan/FragmentRetro.git
cd FragmentRetro
git checkout feat/rl-scoring
pip install -e ".[dev]"

# Run example (text only)
python examples/example_scoring.py

# Run example with route images
python examples/example_scoring.py --draw --outdir my_routes/

# Run tests
python -m pytest tests/ -v -k "not parallel"
```

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `scoring.py` | `is_feasible()`, `compute_score()`, `compute_score_with_dag()` |
| `retro_dag.py` | `RetroNode`, `build_dags()`, `build_best_dag()`, `dag_to_networkx()` |
| `route_visualizer.py` | `draw_route()`, `draw_all_routes()` |
| `bond_feasibility.py` | BRICS bond-type → reliability score lookup |
| `reaction_mapping.py` | BRICS bond-type → named reaction class mapping |
| `batch.py` | `score_batch_binary()`, `score_batch_continuous()`, `score_batch_with_dag()` |
| `substructure_matcher.py` | Cached SMARTS conversion, binary mode matching |
