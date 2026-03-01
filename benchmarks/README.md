# FragmentRetro vs AiZynthFinder Benchmark

Compares FragmentRetro (Tiers 1–3) against AiZynthFinder on the USPTO 190
retrosynthesis test set using an eMolecules ~200K building block stock.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **Solve rate** | Fraction of targets decomposed entirely into purchasable BBs |
| **Wall time** | Per-molecule clock time (mean, median, P95) |
| **Steps** | Total synthesis steps in the best route |
| **LLS** | Longest linear sequence (critical path length) |
| **#BBs** | Number of building blocks needed |

## Quick Start

```bash
# 1. Prepare data
python benchmarks/prepare_data.py \
    --targets /path/to/uspto190.smi \
    --stock /path/to/emolecules_200k.smi \
    --outdir benchmarks/data

# 2. Run FragmentRetro (all tiers)
python benchmarks/run_fragmentretro.py \
    --datadir benchmarks/data

# 3. Run AiZynthFinder
python benchmarks/run_aizynthfinder.py \
    --datadir benchmarks/data \
    --config /path/to/aizynthfinder_config.yml

# 4. Compare results
python benchmarks/analyze_results.py \
    --datadir benchmarks/data
```

## Input Files

### USPTO 190 targets

A text file with one SMILES per line. The standard 190-molecule test set from
retrosynthesis benchmarking literature (Thakkar et al. 2020, Schwaller et al.).

### eMolecules stock

A text file with one SMILES per line (~200K building blocks). `prepare_data.py`
will validate, canonicalise, and create both:
- `stock.smi` — for AiZynthFinder
- `mol_properties.json` — precomputed fingerprints for FragmentRetro

## Scripts

### `prepare_data.py`

Validates and canonicalises target and stock SMILES. Precomputes
FragmentRetro's `mol_properties.json` (pattern fingerprints, heavy atom
counts, ring counts for each BB).

```
Options:
  --targets PATH        USPTO 190 file
  --stock PATH          eMolecules stock file
  --outdir DIR          Output directory (default: benchmarks/data)
  --fp-size INT         Fingerprint size (default: 2048)
  --skip-targets        Reuse existing targets.smi
  --skip-mol-properties Reuse existing mol_properties.json
```

### `run_fragmentretro.py`

Runs FragmentRetro scoring on all targets. Three tiers:

| Tier | Method | Speed | Coverage |
|------|--------|-------|----------|
| 1 | BRICS fragmentation + DAG | Fast | BRICS-representable reactions |
| 2 | BRICS + SMARTS validation | Medium | + validation against 91 literature reactions |
| 3 | SMARTS-only retrosynthesis | Slower | + non-BRICS reactions (Pictet-Spengler, etc.) |

```
Options:
  --datadir DIR           Directory with targets.smi + mol_properties.json
  --tiers 1 2 3           Which tiers to run (default: all)
  --timeout FLOAT         Per-molecule timeout in seconds (default: 60)
  --smarts-depth INT      Tier 3 max tree depth (default: 3)
  --smarts-nodes INT      Tier 3 max nodes explored (default: 500)
  --limit N               Run only first N targets
```

Output: `results_fragmentretro.json`

### `run_aizynthfinder.py`

Runs AiZynthFinder's MCTS tree search on the same targets.

Requires an expansion model — either download the public USPTO model
via `aizynthcli download-public-data`, or provide your own.

```
Options:
  --datadir DIR             Directory with targets.smi + stock.smi
  --config PATH             AiZynthFinder config YAML
  --expansion-model PATH    Model file (.onnx)
  --template-file PATH      Template file (.hdf5)
  --time-limit INT          MCTS time limit per molecule (default: 120)
  --iteration-limit INT     MCTS iterations per molecule (default: 100)
  --limit N                 Run only first N targets
```

Output: `results_aizynthfinder.json`

### `analyze_results.py`

Compares both result files and generates:
- `analysis/summary.tsv` — aggregate comparison table
- `analysis/per_molecule.csv` — side-by-side per-molecule results
- `analysis/solve_rate.png` — solve rate bar chart
- `analysis/time_distribution.png` — timing box plots
- `analysis/time_scatter.png` — FR vs AiZynth per-molecule time scatter
- `analysis/steps_distribution.png` — route length histograms
- `analysis/solve_overlap.png` — Venn-style solve overlap

```
Options:
  --datadir DIR          Directory with result JSON files
  --outdir DIR           Output directory (default: datadir/analysis)
  --no-plots             Skip plot generation
```

## AiZynthFinder Config Example

```yaml
expansion:
  uspto:
    type: template_based
    model: /path/to/uspto_model.onnx
    template: /path/to/templates.hdf5

stock:
  emolecules:
    type: smi
    path: benchmarks/data/stock.smi

search:
  algorithm: mcts
  time_limit: 120
  iteration_limit: 100
  return_first: false
  C: 1.4
  max_transforms: 25
```

## Output JSON Schema

Both runners output JSON with the same per-molecule fields:

```json
{
  "smiles": "CCO",
  "wall_time_s": 0.42,
  "solved": true,
  "score": 0.85,
  "total_steps": 2,
  "longest_linear_sequence": 2,
  "num_building_blocks": 3,
  "timed_out": false,
  "error": null
}
```

This makes it straightforward to extend the analysis or feed results into
your own comparison pipeline.
