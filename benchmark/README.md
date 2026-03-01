# FragmentRetro vs AiZynthFinder Benchmark

Comparative benchmark of retrosynthetic analysis tools on USPTO-190 with a ~200k building block stock.

## Overview

| Metric | What it measures |
|--------|-----------------|
| **Solve rate** | % of targets decomposed into purchasable BBs |
| **Speed** | Wall-clock time per molecule |
| **Route quality** | Number of synthesis steps, LLS, convergence |
| **SMARTS coverage** | Fraction of bonds validated by known reactions (Tier 2) |

## Methods compared

| Method | Description |
|--------|-------------|
| FR Tier 1 binary | BRICS fragmentation, binary solved/unsolved |
| FR Tier 1 continuous | BRICS fragmentation, continuous [0,1] score |
| FR Tier 1 DAG | BRICS + DAG construction with convergence/LLS metrics |
| FR Tier 2 | BRICS + SMARTS validation against 91 literature reactions |
| FR Tier 3 | Standalone SMARTS retrosynthesis (no BRICS dependency) |
| AiZynthFinder | MCTS tree search with neural network expansion policy |

## Quick start

### 1. Install dependencies

```bash
# FragmentRetro (already installed if you're in this repo)
pip install rdkit-pypi matplotlib

# AiZynthFinder
pip install aizynthfinder[all]
download_public_data aizynthfinder
```

### 2. Prepare data

```bash
python benchmark/prepare_data.py \
    --stock-file /path/to/your_200k_stock.smi \
    --output-dir ./benchmark_data
```

This will:
- Download the USPTO-190 target set (or use `--targets-file` for custom targets)
- Convert stock to FragmentRetro `mol_properties.json` format
- Convert stock to AiZynthFinder HDF5/text format
- Write `benchmark_config.json` linking all paths

**Stock file format**: one SMILES per line (optionally tab-separated with names), or `.csv` with a SMILES column.

### 3. Run FragmentRetro benchmark

```bash
# Serial (default)
python benchmark/run_fragmentretro.py \
    --config ./benchmark_data/benchmark_config.json \
    --tiers 1 2 3 \
    --timeout 120

# Parallel with 4 workers (each loads its own CompoundFilter)
python benchmark/run_fragmentretro.py \
    --config ./benchmark_data/benchmark_config.json \
    --workers 4

# Auto-detect CPU count
python benchmark/run_fragmentretro.py \
    --config ./benchmark_data/benchmark_config.json -j 0
```

Options:
- `--tiers 1 2 3` — which tiers to run (1 includes binary+continuous+DAG)
- `--tiers 1b 1c 1d 2 3` — selective tier control
- `--timeout 120` — per-molecule timeout in seconds
- `--max-targets 20` — quick test on subset
- `--max-depth 3` — Tier 3 retrosynthesis depth
- `--max-nodes 500` — Tier 3 search budget
- `--workers N` / `-j N` — parallel workers (0 = all CPUs, 1 = serial)

### 4. Run AiZynthFinder benchmark

```bash
python benchmark/run_aizynthfinder.py \
    --config ./benchmark_data/benchmark_config.json \
    --time-limit 120
```

Options:
- `--time-limit 120` — MCTS search time limit per molecule
- `--policy-model /path/to/model.onnx` — custom expansion model
- `--template-file /path/to/templates.hdf5` — custom templates
- `--aizynthfinder-config config.yml` — use a YAML config file directly

### 5. Analyze results

```bash
python benchmark/analyze_results.py \
    --config ./benchmark_data/benchmark_config.json \
    --output-dir ./benchmark_data/analysis
```

Produces:
- `comparison_summary.json` — aggregate statistics
- `comparison_per_molecule.csv` — per-molecule results
- `comparison_table.tex` — LaTeX table for papers
- `solve_rate_comparison.png` — bar chart
- `speed_comparison.png` — bar chart (log scale)
- `time_scatter.png` — FR vs AiZF per-molecule scatter
- `route_quality_boxplot.png` — steps/BBs distributions
- `solve_by_complexity.png` — solve rate vs molecule size

## Full pipeline (one-liner)

```bash
# Prepare + run all + analyze
python benchmark/prepare_data.py --stock-file stock.smi --output-dir bench && \
python benchmark/run_fragmentretro.py --config bench/benchmark_config.json && \
python benchmark/run_aizynthfinder.py --config bench/benchmark_config.json && \
python benchmark/analyze_results.py --config bench/benchmark_config.json --output-dir bench/analysis
```

## Quick test (10 molecules)

```bash
python benchmark/prepare_data.py --stock-file stock.smi --output-dir bench_test --max-stock 50000
python benchmark/run_fragmentretro.py --config bench_test/benchmark_config.json --max-targets 10 --tiers 1 2
python benchmark/run_aizynthfinder.py --config bench_test/benchmark_config.json --max-targets 10 --time-limit 30
python benchmark/analyze_results.py --config bench_test/benchmark_config.json
```

## Expected output structure

```
benchmark_data/
├── benchmark_config.json       # Links all paths
├── targets.smi                 # USPTO-190 (190 SMILES)
├── mol_properties.json         # FragmentRetro BB fingerprints
├── aizynthfinder_stock.txt     # AiZynthFinder stock (SMILES)
├── aizynthfinder_stock.hdf5    # AiZynthFinder stock (HDF5)
├── results_fragmentretro.json  # FR benchmark output
├── results_aizynthfinder.json  # AiZF benchmark output
└── analysis/
    ├── comparison_summary.json
    ├── comparison_per_molecule.csv
    ├── comparison_table.tex
    ├── solve_rate_comparison.png
    ├── speed_comparison.png
    ├── time_scatter.png
    ├── route_quality_boxplot.png
    └── solve_by_complexity.png
```

## Notes

- **Stock preparation** is the slowest step (~10-30 min for 200k BBs due to fingerprint computation). The resulting `mol_properties.json` can be reused across runs.
- **AiZynthFinder** requires a trained expansion policy model. The default USPTO model from `download_public_data` works for benchmarking but was trained on a different stock, which affects solve rates. For a fair comparison, consider retraining on your specific stock.
- **Timeout matters**: AiZynthFinder's MCTS benefits from longer search times. Set `--time-limit` equal to FragmentRetro's `--timeout` for fair speed comparisons.
- **Tier 3** uses O(b^h) search and is slower than Tiers 1-2. Use `--max-depth 2` and `--max-nodes 200` for faster benchmarks.
