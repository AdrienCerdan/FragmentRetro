# FragmentRetro Intern Onboarding Documentation

Welcome to the **FragmentRetro** project! This document outlines what you need to know to get started, understand our recent methodological improvements (the T1b speedup and T3 SMARTS approach), and run both predictions and benchmarks against AiZynthFinder.

---

## 1. Installation

1. **Install `uv`**:
   We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast, reliable package management.
   ```bash
   pip install uv
   ```

2. **Setup Environment**:
   Clone the repository, switch to the development branch, and create a virtual environment:
   ```bash
   git clone https://github.com/AdrienCerdan/FragmentRetro.git
   cd FragmentRetro
   git checkout feat/rf-scoring
   uv venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   Install FragmentRetro in editable mode along with its core and benchmarking dependencies:
   ```bash
   uv pip install -e .
   uv pip install rdkit-pypi matplotlib aizynthfinder[all]
   download_public_data aizynthfinder
   ```

### 1.5. Production CLI (`retro.py`)
For large-scale or high-performance runs, use the production-grade CLI in `scripts/retro.py`. It supports:
- **Batch inputs**: `.smi` (plain SMILES) or `.csv` (custom column mapping).
- **Parallel processing**: Use `-j N` to spawn multiple workers (each worker loads its own `CompoundFilter`).
- **Flexible output**: Results can be exported to `JSON` or `CSV`.
- **Visualization**: Use `--figures` to automatically generate PNG trees for all solved routes in a directory.

```bash
# Example batch run with 8 workers and strict filters
python scripts/retro.py --input targets.smi --stock mol_properties.json -j 8 --strict
```

---

## 2. Recent Improvements & Methodology

### 2.1. Evolution from Original FragmentRetro
The original FragmentRetro (published in *J. Chem. Theory Comput.* 2026) focused primarily on BRICS-based fragmentation and simple binary/continuous scoring. Our current implementation significantly expands this:
*   **Tiered Methodology**: Instead of a single score, we use T1 (BRICS), T2 (Validation), and T3 (SMARTS) to provide a more nuanced "synthetic accessibility" profile.
*   **Scientific Correctness**: We've added regioselectivity protections and halogen site prioritization (I > Br > Cl) that were absent in the early versions. This prevents the generation of chemically impossible routes.
*   **Quadratic Complexity**: We maintain the landmark quadratic complexity for T1, but T3 allows us to handle non-BRICS reactions (Pictet-Spengler, etc.) which the original method could not solve.

### 2.2. Improved T1b FragmentRetro (Speedup)
FragmentRetro originally suffered from performance bottlenecks when computing multiple tier scores (e.g., T1 binary, T1 continuous, T1 DAG, and T2 validated). The system previously made **redundant calls** (up to 4x) to the core `_run_retro` function for each tier.
* **The T1b Upgrade:** We introduced a unified tier computation function (`compute_all_brics_tiers`). This approach runs the expensive operations—BRICS fragmentation and Substructure Matching against the BB catalog—only **once**. Afterward, it derives all requested tier scores from that shared result.
* **Performance Note:** While the unified call is much faster than the old redundant approach, **maximum throughput is achieved when running T1b (binary) alone**. In this mode, the search can stop as soon as the first solution is found. Including T1c (continuous), T1d (DAG), or T2 (validated) requires full enumeration and additional graph/SMARTS analysis, which bounds the overall performance.
* **Behavior:** The behavior and rules are identical to the original FR; the only difference is the elimination of redundancy.

### 2.3. Caching & Performance Optimization
To achieve "real-time" performance even with complex SMARTS patterns, we've implemented a multi-layered caching strategy:
-   **LRU Caching**: We use `@lru_cache` extensively (up to 16,384 entries) for expensive RDKit operations like SMILES parsing, SMARTS compilation, and `AddHs` molecule preparation. This ensures that the same building block or reaction pattern is never processed twice.
-   **Purchasability Cache**: Within the T3 engine (`SmartsRetrosynthesis`), we maintain a `_purchasable_cache`. Since the same intermediate fragments often appear across different branches of the tree, this dictionary lookup replaces hundreds of redundant `CompoundFilter` bitset queries per molecule.
-   **Worker-Level Globals**: In parallel runs, each worker initializes its `CompoundFilter` once in global memory, minimizing IO and deserialization overhead between tasks.

### 2.2. T3 Methodology with SMARTS Patterns
While T1 relies on BRICS fragmentation (rule-based), **T3 is our standalone SMARTS-based retrosynthesis tier**. It can cover reactions that BRICS cannot handle (e.g., Pictet-Spengler, Fischer indole). T3 uses the same compound filter (BB catalog) as the rest of the tiers but leverages advanced SMARTS patterns for retrosynthetic disconnections.

Key methodological highlights of T3:
* **Hierarchical Halogen Selectivity:** We've implemented tiered Pd-catalyzed reaction groups matching true chemical reactivity (I > Br > Cl) using recursive SMARTS logic. This ensures that when a highly reactive site (e.g., an Iodide) is present, the presence of a lower priority halide on the same molecule doesn't confound the site prioritization.
* **Universal Regioselectivity Protection:** Reaction exclusions are strictly standardized for building blocks containing multiple identical aromatic halides (e.g., `[c]-[I].[c]-[I]`). By filtering these through `exclusions_on_any_bb`, the system prevents ambiguous regiochemical outcomes, keeping the generated routes scientifically realistic.

*(Note: T3 currently relies on a best-first DFS. True priority-queue based search is a future horizon.)*

---

## 3. Running Predictions and Benchmarks

The project comes with built-in scripts to test and benchmark FragmentRetro against AiZynthFinder. A general benchmarking sequence typically involves preparing the data, running the tools, and analyzing the results.

### General Benchmark Pipeline

1. **Data Preparation:** You must map the building block stocks for both tools. *(Note: The AiZynthFinder default model uses a USPTO-trained stock which might have lower solve rates specifically for our small 13k-200k stocks).*
   ```bash
   python benchmark/prepare_data.py \
       --stock-file /path/to/stock.smi \
       --output-dir ./benchmark_test \
       --max-stock 50000
   ```
2. **Run FragmentRetro (Tiers 1, 2, 3):**
   ```bash
   # Multi-tier run (comprehensive analysis)
   python benchmark/run_fragmentretro.py \
       --config ./benchmark_test/benchmark_config.json \
       --max-targets 10 \
       --tiers 1 2 3 \
       --workers 4

   # T1b only run (maximum throughput / binary solve rate)
   # Use '--tiers 1b' to run the fastest binary fragmentation check only.
   python benchmark/run_fragmentretro.py \
       --config ./benchmark_test/benchmark_config.json \
       --tiers 1b -j 8
   ```
3. **Run AiZynthFinder:**
   ```bash
   python benchmark/run_aizynthfinder.py \
       --config ./benchmark_test/benchmark_config.json \
       --max-targets 10 \
       --time-limit 120
   ```
4. **Analyze the Results:**
   Generates scatter plots, solve-rate comparisons, and latex tables to evaluate strengths and weaknesses.
   ```bash
   python benchmark/analyze_results.py \
       --config ./benchmark_test/benchmark_config.json \
       --output-dir ./benchmark_test/analysis
   ```

### Strengths & Weaknesses (What to look out for in analysis)
* **FragmentRetro:** Excels rapidly against small/custom building block stocks because its explicit rule-based fragmentation explicitly closes routes onto known matching fragments. You will see significantly higher solve rates on specific stocks.
* **AiZynthFinder:** Provides broader general USPTO coverage but traditionally struggles out-of-the-box when restricted strictly to small, specific stock dictionaries unless retrained.
