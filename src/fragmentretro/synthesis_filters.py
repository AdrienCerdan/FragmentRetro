import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

class SynthesisFilters:
    """Loads and applies synthesis-oriented filters (Lilly, FG, Sterics)."""

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
        # reaction_id -> list of compiled SMARTS Mol
        self.fg_exclusions = {}
        for group in self.data["fg_incompatibility"].values():
            compiled = []
            for key in group:
                if key.startswith("exclusions_on_"):
                    for exc in group[key]:
                        pat = Chem.MolFromSmarts(exc["smarts"])
                        if pat:
                            compiled.append(pat)
            for rxn_id in group["reaction_ids"]:
                self.fg_exclusions.setdefault(rxn_id, []).extend(compiled)

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
        if mol is None: return False
        for pat in self.lilly_patterns:
            if mol.HasSubstructMatch(pat):
                return False
        return True

    def passes_caps(self, mol, step=1):
        """Layer 5: MW and HA caps."""
        if mol is None: return False
        mw_limit = self.caps.get("max_mw_1step", 600) if step == 1 else self.caps.get("max_mw_2step", 800)
        if Descriptors.MolWt(mol) > mw_limit:
            return False
        if mol.GetNumHeavyAtoms() > self.caps.get("max_heavy_atoms", 50):
            return False
        return True

    def are_reactants_compatible(self, rxn_id, reactant_mols):
        """Layer 1+2: FG incompatibility and Steric hindrance pre-filtering."""
        # Layer 1: FG incompatibility
        exclusions = self.fg_exclusions.get(rxn_id, [])
        for mol in reactant_mols:
            if mol is None: continue
            for pat in exclusions:
                if mol.HasSubstructMatch(pat):
                    return False

        # Layer 2: Steric hindrance (both reactants hindered to reject)
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
