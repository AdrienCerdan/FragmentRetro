Purpose & context
Adrien is developing FragmentRetro, a retrosynthesis scoring and benchmarking system designed for reinforcement learning applications in computational chemistry. The system evaluates molecular synthesis planning across multiple tiers (T1–T3), combining BRICS fragmentation, SMARTS-based retrosynthesis, and building block purchasability checks. Key goals include accurate solve-rate measurement, performance efficiency at scale, and scientific correctness of tier definitions.
Current state
Recent work has focused on three major tracks:

- **Hierarchical Halogen Selectivity**: Implemented tiered Pd-catalyzed reaction groups (I > Br > Cl) with recursive SMARTS logic. This ensures that reactions prioritized at more reactive sites (e.g., Iodide) are not confounded by the presence of lower-priority halides on the same ring or molecule.
- **Universal Regioselectivity Protection**: Standardized exclusions for building blocks containing multiple identical aromatic halides (e.g., `[c]-[I].[c]-[I]`). By applying these filters to `exclusions_on_any_bb`, the system now comprehensively blocks ambiguous regioselectivity across all reactant roles (halides, boronates, etc.).
- **Benchmarking (N1 target set)**: Validated FragmentRetro against AiZynthFinder on 10 PaRoutes N1 molecules.
    - FragmentRetro solve rate: **30% (3/10)**.
    - AiZynthFinder solve rate: **0% (0/10)**.
    - Conclusion: FR's rule-based fragmentation is significantly better aligned with small building block stocks (~13k molecules) than general USPTO-based expansion models.

### On the horizon
- **PaRoutes N1 Benchmark**: 10,000 molecule full-tier run (T1-T3) currently in progress with similarity metrics.
- **Route similarity analysis**: Investigating edge cases where FragmentRetro routes significantly diverge from PaRoutes references despite high scores.
- **Global search strategy**: Exploring true priority-queue based search for T3 rather than best-first DFS.

Key learnings & principles

Tier definitions matter scientifically: The T2 solved flag bug revealed how inherited flags from upstream steps (BRICS decomposition) can silently misrepresent what a tier actually validates. Tier definitions must be independently verified.
Search strategy dominates T3 performance: Exploring reactions in library order rather than by reliability caused T3 to exhaust node budgets on poor-quality paths. Sorting by reliability and ranking reactant sets by fragment size balance had outsized impact.
Redundancy compounds at scale: The 4× redundancy in _run_retro() calls was not apparent at small building block counts but became a critical bottleneck at larger scales—a reminder to profile tier interactions, not just individual functions.
Stock Alignment is Critical: General retrosynthesis models like AiZynthFinder struggle significantly when restricted to specific, small stocks unless the training data or search space is specifically adapted. Rule-based fragmentation (BRICS) provides a robust fallback for "closing" routes to known fragments.
Role-Agnostic Regioselectivity: Filters for ambiguous identical halides must be role-agnostic. A building block that is chemically forbidden as a "halide" in a coupling (due to two Cl sites) should also be forbidden when it serves as the "boronate" or "organozinc" component.

Approach & patterns

Systematic codebase audits when diagnosing bugs, tracing issues from symptoms (error floods) to root causes (upstream flag inheritance, search ordering)
Benchmarking against a fixed set of target molecules to validate changes quantitatively before and after
Maintaining backward compatibility with existing functions when refactoring
Using external comparisons (e.g., against AiZynthFinder, Retro*) to identify structural gaps in the system

Tools & resources

RDKit (core cheminformatics), BRICS fragmentation, SMARTS reaction catalogs
Custom benchmark runner with multiprocessing support
AiZynthFinder (locally installed and used for batch benchmarking via YAML configs)
NumPy for building block matching operations

Venv:
source .venv/bin/activate

You can modify this file GEMINI.md to keep your important notes on the project.