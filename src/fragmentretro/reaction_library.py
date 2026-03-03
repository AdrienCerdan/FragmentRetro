"""Reaction library for SMARTS-based synthesis validation and retrosynthesis.

Provides:
  - Reaction: dataclass representing a single named reaction with SMARTS
  - ReactionLibrary: loadable, filterable collection of Reaction objects
  - Built-in catalogs: Hartenfeller (58 reactions), eXplore Cookbook (~35)
  - Custom user reaction loading from JSON files
  - SMARTS-based product matching for DAG bond validation
  - Reverse (retrosynthetic) SMARTS application
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import re

from rdkit import Chem
from rdkit.Chem import AllChem

from fragmentretro.utils.logging_config import logger

# Path to bundled reaction catalogs
_DATA_DIR = Path(__file__).parent / "data"
HARTENFELLER_PATH = _DATA_DIR / "hartenfeller_reactions.json"
EXPLORE_PATH = _DATA_DIR / "explore_reactions.json"


def _replace_dummy_atoms(smiles: str) -> str | None:
    """Replace dummy atoms in a SMILES with hydrogen and re-canonicalize.

    Dummy atoms ('*') arise when reverse SMARTS application cannot resolve
    ambiguous atom patterns (e.g., [Cl,Br,I] becomes '*').  Replacing with
    hydrogen gives the molecular core, which is appropriate for substructure-
    based purchasability checks against a building-block catalog.

    Fragments with fewer than 2 heavy atoms after replacement are discarded
    as artifacts of spurious SMARTS matches (e.g., Suzuki matching at an
    alkyl–aryl bond yields '*[C]' → '[CH]', which is chemically nonsensical
    as a building block).

    Args:
        smiles: SMILES that may contain '*' atoms.

    Returns:
        Canonical SMILES without dummy atoms, or None if the result is
        chemically invalid or too small to be a meaningful building block.
    """
    # Replace bracketed forms [1*], [*] and bare *
    cleaned = re.sub(r"\[\d*\*\]", "[H]", smiles)
    cleaned = re.sub(r"(?<!\[)\*(?!\])", "[H]", cleaned)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    # Discard single-atom fragments (artifacts of wrong SMARTS match sites)
    if mol.GetNumHeavyAtoms() < 2:
        return None
    return Chem.MolToSmiles(mol)


@dataclass(frozen=True)
class Reaction:
    """A single named chemical reaction with SMARTS pattern.

    Attributes:
        id: Unique identifier (e.g., 'hartenfeller_31').
        name: Human-readable name (e.g., 'Suzuki').
        reaction_class: Broad class: 'coupling', 'substitution', 'ring_closure', 'multicomponent'.
        product_class: What the reaction produces (e.g., 'biaryl', 'amide').
        smarts_forward: Forward SMARTS (reactants >> products).
        num_reactants: Number of reactant components.
        reliability: Estimated reliability score in [0, 1].
        source: Catalog source (e.g., 'hartenfeller', 'explore', 'user').
        notes: Additional notes or caveats.
    """

    id: str
    name: str
    reaction_class: str
    product_class: str
    smarts_forward: str
    num_reactants: int = 2
    reliability: float = 0.8
    source: str = "unknown"
    notes: str = ""

    @lru_cache(maxsize=1)
    def rdkit_rxn(self) -> Optional[AllChem.ChemicalReaction]:
        """Compile the forward SMARTS into an RDKit reaction object (cached)."""
        try:
            rxn = AllChem.ReactionFromSmarts(self.smarts_forward)
            if rxn is not None:
                rxn.Initialize()
            return rxn
        except Exception as e:
            logger.warning(f"[ReactionLibrary] Failed to parse SMARTS for {self.id}: {e}")
            return None

    @lru_cache(maxsize=1)
    def reverse_rdkit_rxn(self) -> Optional[AllChem.ChemicalReaction]:
        """Compile the reverse SMARTS (product >> reactants) for retrosynthetic use."""
        try:
            parts = self.smarts_forward.split(">>")
            if len(parts) != 2:
                return None
            reverse_smarts = f"{parts[1]}>>{parts[0]}"
            rxn = AllChem.ReactionFromSmarts(reverse_smarts)
            if rxn is not None:
                rxn.Initialize()
            return rxn
        except Exception as e:
            logger.debug(f"[ReactionLibrary] Failed to parse reverse SMARTS for {self.id}: {e}")
            return None

    @lru_cache(maxsize=1)
    def product_query(self) -> Optional[Chem.Mol]:
        """Compile the product side of the SMARTS into a query mol (cached).

        Used for fast substructure pre-filtering: check if a target molecule
        contains the functional groups required by this reaction.
        """
        product_smarts = self.smarts_forward.split(">>")[-1]
        try:
            return Chem.MolFromSmarts(product_smarts)
        except Exception:
            return None

    def matches_product(self, product_mol: Chem.Mol) -> bool:
        """Check if a product molecule could have been made by this reaction.

        Uses a pre-compiled product SMARTS query for fast substructure matching.
        """
        query = self.product_query()
        if query is None:
            return False
        try:
            return product_mol.HasSubstructMatch(query)
        except Exception:
            return False

    @lru_cache(maxsize=4096)
    def apply_reverse(self, product_smiles: str) -> list[tuple[str, ...]]:
        """Apply the reaction in reverse to get possible reactant sets.

        Dummy atoms ('*') produced by unresolved SMARTS patterns are replaced
        with hydrogen and the result is re-canonicalized, so that downstream
        purchasability checks work correctly.

        Args:
            product_smiles: SMILES of the product molecule.

        Returns:
            List of tuples of reactant SMILES. Empty if no match.
        """
        rxn = self.reverse_rdkit_rxn()
        if rxn is None:
            return []

        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return []

        try:
            product_sets = rxn.RunReactants((mol,))
        except Exception:
            return []

        results: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()

        for products in product_sets:
            reactant_smiles: list[str] = []
            valid = True
            for p in products:
                try:
                    Chem.SanitizeMol(p)
                    smi = Chem.MolToSmiles(p)
                    if not smi:
                        valid = False
                        break
                    # Replace dummy atoms produced by unresolved SMARTS patterns
                    # (e.g., [Cl,Br,I] -> *) with hydrogen, then re-canonicalize
                    if '*' in smi:
                        smi = _replace_dummy_atoms(smi)
                        if smi is None:
                            valid = False
                            break
                    reactant_smiles.append(smi)
                except Exception:
                    valid = False
                    break

            if valid and reactant_smiles:
                key = tuple(sorted(reactant_smiles))
                if key not in seen:
                    seen.add(key)
                    results.append(tuple(reactant_smiles))

        return results


def _strip_dummy_atoms(smiles: str) -> Optional[str]:
    """Strip dummy atoms (atomic number 0) from a SMILES and return canonical core.

    Handles both BRICS dummy atoms ([16*], [1*], etc.) and SMARTS dummy atoms (*).
    Returns None if the result is empty or invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Check if there are any dummy atoms to strip
    dummy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_idxs:
        # No dummies — return canonical SMILES directly
        return Chem.MolToSmiles(mol)

    from rdkit.Chem import RWMol
    rw = RWMol(mol)
    for idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(idx)

    clean = rw.GetMol()
    if clean.GetNumAtoms() == 0:
        return None

    try:
        Chem.SanitizeMol(clean)
        return Chem.MolToSmiles(clean)
    except Exception:
        return None


class ReactionLibrary:
    """A filterable collection of Reaction objects.

    Supports loading from built-in catalogs and custom JSON files,
    filtering by name/class/reaction set, and SMARTS-based product matching.

    Usage:
        # Load all built-in reactions
        lib = ReactionLibrary.default()

        # Load only Hartenfeller
        lib = ReactionLibrary.from_hartenfeller()

        # Load with custom additions
        lib = ReactionLibrary.default()
        lib.load_json("my_reactions.json", source="custom")

        # Filter to specific reactions
        subset = lib.filter(names=["Suzuki", "amide_coupling"])
        subset = lib.filter(classes=["coupling", "ring_closure"])
        subset = lib.filter(min_reliability=0.8)
    """

    def __init__(self, reactions: list[Reaction] | None = None):
        self._reactions: dict[str, Reaction] = {}
        if reactions:
            for rxn in reactions:
                self._reactions[rxn.id] = rxn

    @property
    def reactions(self) -> list[Reaction]:
        return list(self._reactions.values())

    @property
    def reaction_ids(self) -> list[str]:
        return list(self._reactions.keys())

    @property
    def reaction_names(self) -> list[str]:
        return sorted({r.name for r in self._reactions.values()})

    @property
    def reaction_classes(self) -> list[str]:
        return sorted({r.reaction_class for r in self._reactions.values()})

    def __len__(self) -> int:
        return len(self._reactions)

    def __contains__(self, reaction_id: str) -> bool:
        return reaction_id in self._reactions

    def __getitem__(self, reaction_id: str) -> Reaction:
        return self._reactions[reaction_id]

    def get(self, reaction_id: str) -> Optional[Reaction]:
        return self._reactions.get(reaction_id)

    def add(self, reaction: Reaction) -> None:
        """Add a single reaction to the library."""
        self._reactions[reaction.id] = reaction

    def remove(self, reaction_id: str) -> None:
        """Remove a reaction by ID."""
        self._reactions.pop(reaction_id, None)

    # --- Loading ---

    def load_json(self, path: str | Path, source: str = "user") -> int:
        """Load reactions from a JSON file.

        Expected format: {"reactions": [{"id": ..., "name": ..., "class": ...,
        "product_class": ..., "smarts_forward": ..., ...}, ...]}

        Args:
            path: Path to JSON file.
            source: Source label for loaded reactions.

        Returns:
            Number of reactions loaded.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        count = 0
        for entry in data.get("reactions", []):
            try:
                rxn = Reaction(
                    id=entry["id"],
                    name=entry["name"],
                    reaction_class=entry.get("class", "unknown"),
                    product_class=entry.get("product_class", "unknown"),
                    smarts_forward=entry["smarts_forward"],
                    num_reactants=entry.get("num_reactants", 2),
                    reliability=entry.get("reliability", 0.8),
                    source=source,
                    notes=entry.get("notes", ""),
                )
                self._reactions[rxn.id] = rxn
                count += 1
            except (KeyError, TypeError) as e:
                logger.warning(f"[ReactionLibrary] Skipping malformed entry in {path}: {e}")

        logger.info(f"[ReactionLibrary] Loaded {count} reactions from {path} (source={source})")
        return count

    @classmethod
    def from_hartenfeller(cls) -> ReactionLibrary:
        """Load only the Hartenfeller 58 reactions."""
        lib = cls()
        lib.load_json(HARTENFELLER_PATH, source="hartenfeller")
        return lib

    @classmethod
    def from_explore(cls) -> ReactionLibrary:
        """Load only the eXplore Cookbook reactions."""
        lib = cls()
        lib.load_json(EXPLORE_PATH, source="explore")
        return lib

    @classmethod
    def default(cls) -> ReactionLibrary:
        """Load both Hartenfeller + eXplore catalogs."""
        lib = cls()
        lib.load_json(HARTENFELLER_PATH, source="hartenfeller")
        lib.load_json(EXPLORE_PATH, source="explore")
        return lib

    # --- Filtering ---

    def filter(
        self,
        *,
        names: list[str] | None = None,
        ids: list[str] | None = None,
        classes: list[str] | None = None,
        product_classes: list[str] | None = None,
        sources: list[str] | None = None,
        min_reliability: float = 0.0,
        max_reactants: int | None = None,
    ) -> ReactionLibrary:
        """Return a new ReactionLibrary with only matching reactions.

        All filters are AND-combined. Name matching is case-insensitive and
        supports partial matching (e.g., 'suzuki' matches 'Suzuki_biaryl').

        Args:
            names: Filter by reaction name (case-insensitive, partial match).
            ids: Filter by exact reaction ID.
            classes: Filter by reaction class.
            product_classes: Filter by product class.
            sources: Filter by source.
            min_reliability: Minimum reliability score.
            max_reactants: Maximum number of reactants.

        Returns:
            New filtered ReactionLibrary.
        """
        filtered: list[Reaction] = []
        names_lower = [n.lower() for n in names] if names else None

        for rxn in self._reactions.values():
            if ids is not None and rxn.id not in ids:
                continue
            if names_lower is not None:
                rxn_name_lower = rxn.name.lower()
                if not any(n in rxn_name_lower for n in names_lower):
                    continue
            if classes is not None and rxn.reaction_class not in classes:
                continue
            if product_classes is not None and rxn.product_class not in product_classes:
                continue
            if sources is not None and rxn.source not in sources:
                continue
            if rxn.reliability < min_reliability:
                continue
            if max_reactants is not None and rxn.num_reactants > max_reactants:
                continue
            filtered.append(rxn)

        return ReactionLibrary(filtered)

    # --- SMARTS matching ---

    def find_matching_reactions(
        self,
        product_smiles: str,
        top_k: int = 5,
    ) -> list[tuple[Reaction, float]]:
        """Find reactions whose product SMARTS matches the given molecule.

        Returns reactions sorted by reliability score (descending).

        Args:
            product_smiles: SMILES string of the product.
            top_k: Maximum number of results.

        Returns:
            List of (Reaction, reliability_score) tuples.
        """
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return []

        matches: list[tuple[Reaction, float]] = []
        for rxn in self._reactions.values():
            if rxn.matches_product(mol):
                matches.append((rxn, rxn.reliability))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def validate_bond_disconnection(
        self,
        parent_smiles: str,
        child_smiles_list: list[str],
        brics_bond_type: tuple[str, str] | None = None,
    ) -> tuple[bool, Optional[Reaction], float]:
        """Validate whether a retrosynthetic disconnection matches any known reaction.

        Tests if any reaction in the library, applied in reverse to parent_smiles,
        could produce fragments matching child_smiles_list.

        Handles dummy atoms in both BRICS fragment SMILES (e.g., [16*]c1ccccc1)
        and reverse reaction products (e.g., *c1ccccc1) by comparing molecular
        cores after stripping all dummy atoms.

        Args:
            parent_smiles: SMILES of the molecule being disconnected.
            child_smiles_list: SMILES of the resulting fragments.
            brics_bond_type: Optional BRICS bond type hint for faster matching.

        Returns:
            (is_valid, best_matching_reaction, reliability_score)
        """
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            return False, None, 0.0

        child_cores = set()
        for cs in child_smiles_list:
            core = _strip_dummy_atoms(cs)
            if core:
                child_cores.add(core)

        if not child_cores:
            return False, None, 0.0

        best_rxn: Optional[Reaction] = None
        best_score = 0.0

        # Pre-filter: only reactions whose product SMARTS matches parent
        for rxn in self._reactions.values():
            if not rxn.matches_product(parent_mol):
                continue

            # Try reverse application
            reactant_sets = rxn.apply_reverse(parent_smiles)
            for reactants in reactant_sets:
                reactant_cores = set()
                for r in reactants:
                    core = _strip_dummy_atoms(r)
                    if core:
                        reactant_cores.add(core)

                # Check overlap on molecular cores
                overlap = child_cores & reactant_cores
                if overlap:
                    coverage = len(overlap) / len(child_cores)
                    score = rxn.reliability * coverage
                    if score > best_score:
                        best_score = score
                        best_rxn = rxn

        if best_rxn is not None:
            return True, best_rxn, best_score

        return False, None, 0.0

    def summary(self) -> str:
        """Human-readable summary of the library contents."""
        lines = [f"ReactionLibrary: {len(self)} reactions"]
        for cls in self.reaction_classes:
            count = sum(1 for r in self._reactions.values() if r.reaction_class == cls)
            lines.append(f"  {cls}: {count}")
        sources = sorted({r.source for r in self._reactions.values()})
        for src in sources:
            count = sum(1 for r in self._reactions.values() if r.source == src)
            lines.append(f"  source={src}: {count}")
        return "\n".join(lines)
