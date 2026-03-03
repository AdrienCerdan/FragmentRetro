Purpose & context
Adrien is developing FragmentRetro, a retrosynthesis scoring and benchmarking system designed for reinforcement learning applications in computational chemistry. The system evaluates molecular synthesis planning across multiple tiers (T1–T3), combining BRICS fragmentation, SMARTS-based retrosynthesis, and building block purchasability checks. Key goals include accurate solve-rate measurement, performance efficiency at scale, and scientific correctness of tier definitions.
Current state
Recent work has focused on two major tracks:

- **Clean Logging**: Centralized RDKit and internal `fragmentretro` log suppression in `logging_config.py`. This eliminated verbose valence/aromaticity warnings and initialization noise, facilitating large-scale benchmark monitoring.
- **Scaling Optimizations**: Implemented cross-molecule reaction caching in T3 and molecular object caching in `SubstructureMatcher`, delivering significant overhead reduction for deep search trees.
- **Non-Greedy BRICS**: Tier 2 now evaluates multiple decompositions (up to 10 solutions) to ensure optimal SMARTS validation coverage, moving beyond the greedy "shortest-path" approach.

### On the horizon
- **PaRoutes N1 Benchmark**: 10,000 molecule full-tier run (T1-T3) currently in progress with similarity metrics.
- **Route similarity analysis**: Investigating edge cases where FragmentRetro routes significantly diverge from PaRoutes references despite high scores.
- **Global search strategy**: Exploring true priority-queue based search for T3 rather than best-first DFS.

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