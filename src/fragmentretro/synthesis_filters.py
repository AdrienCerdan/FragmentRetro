"""Synthesis-oriented filters for chemical space estimation and retrosynthesis.

Implements:
  - Lilly instability SMARTS (reactive/unstable functional groups)
  - FG incompatibility (reaction-specific exclusions)
  - Role-aware filtering (halide-BB vs boronic-acid-BB specific rules)
  - Steric hindrance (both reactants hindered → reject)
  - MW / heavy-atom caps
"""

import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors


# Role-detection SMARTS: identify which reactant plays which role
_ROLE_DETECTORS = {
    "halide_bb": Chem.MolFromSmarts("[#6;a]-[Br,I]"),      # ArBr or ArI (Suzuki-active)
    "boronic_bb": Chem.MolFromSmarts("[#6;a]-[B](O)O"),     # ArB(OH)2
    "acid_bb": Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]"), # -COOH
    "amine_bb": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),# primary/secondary amine
}


class SynthesisFilters:
    """Loads and applies synthesis-oriented filters (Lilly, FG, Sterics).

    Role-aware filtering: exclusions like 'exclusions_on_halide_bb' are
    only applied to reactants that actually carry the matching functional
    group (ArBr/ArI for halide_bb). This prevents false positives where
    a non-halide reactant gets rejected by halide-specific rules.
    """

    def __init__(self, json_path=None):
        if json_path is None:
            json_path = Path(__file__).resolve().parent / "data" / "synthesis_filters.json"

        with open(json_path) as f:
            self.data = json.load(f)

        # Compile Lilly instability patterns
        self.lilly_patterns = []
        for entry in self.data["lilly_instability_smarts"]:
            pat = Chem.MolFromSmarts(entry["smarts"])
            if pat:
                self.lilly_patterns.append(pat)

        # Compile FG incompatibility patterns per reaction_id
        # Structure: reaction_id -> {"any": [...], "halide_bb": [...], ...}
        self.fg_exclusions = {}
        for group in self.data["fg_incompatibility"].values():
            any_compiled = []
            role_compiled = {}  # role_name -> [patterns]

            for exc in group.get("exclusions_on_any_bb", []):
                pat = Chem.MolFromSmarts(exc["smarts"])
                if pat:
                    any_compiled.append(pat)

            # Compile role-specific exclusions
            for key, entries in group.items():
                if key.startswith("exclusions_on_") and key != "exclusions_on_any_bb":
                    role_name = key.replace("exclusions_on_", "")  # e.g. "halide_bb"
                    role_pats = []
                    for exc in entries:
                        pat = Chem.MolFromSmarts(exc["smarts"])
                        if pat:
                            role_pats.append(pat)
                    if role_pats:
                        role_compiled[role_name] = role_pats

            for rxn_id in group.get("reaction_ids", []):
                self.fg_exclusions.setdefault(rxn_id, {"any": [], "roles": {}})
                self.fg_exclusions[rxn_id]["any"].extend(any_compiled)
                for role_name, pats in role_compiled.items():
                    self.fg_exclusions[rxn_id]["roles"].setdefault(role_name, []).extend(pats)

        # Compile steric hindrance patterns
        self.steric_patterns = []
        steric = self.data.get("steric_hindrance", {})
        for entry in steric.get("patterns", []):
            pat = Chem.MolFromSmarts(entry["smarts"])
            if pat:
                self.steric_patterns.append(pat)

        self.caps = self.data.get("product_caps", {})

    def passes_instability_filter(self, mol):
        """Layer 4: Lilly instability SMARTS."""
        if mol is None:
            return False
        for pat in self.lilly_patterns:
            if mol.HasSubstructMatch(pat):
                return False
        return True

    def passes_caps(self, mol, step=1):
        """Layer 5: MW and HA caps."""
        if mol is None:
            return False
        mw_limit = self.caps.get("max_mw_1step", 600) if step == 1 else self.caps.get("max_mw_2step", 800)
        if Descriptors.MolWt(mol) > mw_limit:
            return False
        if mol.GetNumHeavyAtoms() > self.caps.get("max_heavy_atoms", 50):
            return False
        return True

    @staticmethod
    def _detect_role(mol, role_name):
        """Check if a molecule matches the role pattern (e.g. 'halide_bb')."""
        detector = _ROLE_DETECTORS.get(role_name)
        if detector is None:
            return False
        return mol.HasSubstructMatch(detector)

    def are_reactants_compatible(self, rxn_id, reactant_mols):
        """Layer 1+2: FG incompatibility and Steric hindrance pre-filtering.

        Applies exclusions_on_any_bb to ALL reactants, but
        role-specific exclusions (e.g. exclusions_on_halide_bb)
        only to reactants that actually carry that functional group.
        """
        exclusion_data = self.fg_exclusions.get(rxn_id, {"any": [], "roles": {}})

        # Layer 1a: Generic exclusions (apply to every reactant)
        for mol in reactant_mols:
            if mol is None:
                continue
            for pat in exclusion_data["any"]:
                if mol.HasSubstructMatch(pat):
                    return False

        # Layer 1b: Role-specific exclusions
        for role_name, role_pats in exclusion_data.get("roles", {}).items():
            for mol in reactant_mols:
                if mol is None:
                    continue
                # Only apply these patterns if this reactant matches the role
                if not self._detect_role(mol, role_name):
                    continue
                for pat in role_pats:
                    if mol.HasSubstructMatch(pat):
                        return False

        # Layer 2: Steric hindrance (both reactants hindered → reject)
        if self.steric_patterns and len(reactant_mols) == 2:
            both_hindered = True
            for mol in reactant_mols:
                if mol is None:
                    both_hindered = False
                    break
                is_hindered = any(mol.HasSubstructMatch(sp) for sp in self.steric_patterns)
                if not is_hindered:
                    both_hindered = False
                    break
            if both_hindered:
                return False

        return True
