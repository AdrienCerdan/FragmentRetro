Purpose & context
Adrien is developing FragmentRetro, a retrosynthesis scoring and benchmarking system designed for reinforcement learning applications in computational chemistry. The system evaluates molecular synthesis planning across multiple tiers (T1–T3), combining BRICS fragmentation, SMARTS-based retrosynthesis, and building block purchasability checks. Key goals include accurate solve-rate measurement, performance efficiency at scale, and scientific correctness of tier definitions.
Current state
Recent work has focused on two major tracks:

Performance optimization: Identified and resolved critical redundancy where _run_retro() was called independently per tier. Implemented a unified compute_all_brics_tiers() function to share results across tiers, delivering ~2.8× speedup on BRICS tiers. Optimized has_match() with early-exit logic.
Bug fixes and correctness: Resolved dummy atom (*) contamination in SMILES from reverse SMARTS reactions, fixed broken SMARTS patterns in the reaction catalog, corrected a silent failure in constraints filtering, and addressed AiZynthFinder API compatibility issues. Critically, corrected T2's "solved" flag to require full SMARTS validation coverage rather than just BRICS decomposition success—a previously misleading definition that inflated apparent T2 performance.
T3 overhaul: Rewrote the SmartsRetrosynthesis engine with pre-filtering, purchasability caching, best-first search by reaction reliability, early termination, and increased node budget. Solve rate improved from ~30% to ~80% with a major reduction in per-molecule processing time.
Parallelization: Added multiprocessing support to the benchmark runner (--workers/-j flags) with worker initialization to avoid re-serializing large numpy arrays.

On the horizon
Three concrete improvement areas identified for FragmentRetro's next iteration:

Exploring multiple BRICS decompositions instead of greedily selecting the shortest
Implementing functional group pre-filtering for T3 reactions
Adding stock-aware pruning to avoid expanding fragments that cannot match available building blocks

Open questions remain around the decoupled scoring logic causing inconsistencies between tier-level and route-level solved flags in detailed results.
Key learnings & principles

Tier definitions matter scientifically: The T2 solved flag bug revealed how inherited flags from upstream steps (BRICS decomposition) can silently misrepresent what a tier actually validates. Tier definitions must be independently verified.
Search strategy dominates T3 performance: Exploring reactions in library order rather than by reliability caused T3 to exhaust node budgets on poor-quality paths. Sorting by reliability and ranking reactant sets by fragment size balance had outsized impact.
Redundancy compounds at scale: The 4× redundancy in _run_retro() calls was not apparent at small building block counts but became a critical bottleneck at larger scales—a reminder to profile tier interactions, not just individual functions.
SMARTS covers structurally harder molecules: T3 SMARTS retrosynthesis achieves higher coverage than BRICS-only approaches by handling complex reactions (e.g., Pictet-Spengler, Fischer indole) that BRICS cannot represent, though at lower quality scores—expected given the harder molecular targets.

Approach & patterns

Systematic codebase audits when diagnosing bugs, tracing issues from symptoms (error floods) to root causes (upstream flag inheritance, search ordering)
Benchmarking against a fixed set of target molecules to validate changes quantitatively before and after
Maintaining backward compatibility with existing functions when refactoring
Using external comparisons (e.g., against AiZynthFinder, Retro*) to identify structural gaps in the system

Tools & resources

RDKit (core cheminformatics), BRICS fragmentation, SMARTS reaction catalogs
Custom benchmark runner with multiprocessing support
AiZynthFinder (referenced for API compatibility and comparative benchmarking)
NumPy for building block matching operations

Venv:
source .venv/bin/activate

You can modify this file GEMINI.md to keep your important notes on the project.