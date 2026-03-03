import json
from rdkit import Chem

with open("benchmark_data_n1/targets.smi") as f:
    target_smiles = [line.strip() for line in f if line.strip()]

with open("data/paroutes/n1-routes.json") as f:
    ref = json.load(f)

ref_can = set()
for r in ref:
    m = Chem.MolFromSmiles(r["smiles"])
    if m:
        ref_can.add(Chem.MolToSmiles(m, isomericSmiles=False))

found = 0
not_found = 0
for ts in target_smiles:
    m = Chem.MolFromSmiles(ts)
    if m:
        tcan = Chem.MolToSmiles(m, isomericSmiles=False)
        if tcan in ref_can:
            found += 1
        else:
            not_found += 1

print(f"Total targets: {len(target_smiles)}")
print(f"Found in ref (isomeric=False): {found}")
print(f"Not found in ref: {not_found}")
