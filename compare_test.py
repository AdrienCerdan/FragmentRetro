import json
from rdkit import Chem

with open("test_benchmark.json") as f:
    res = json.load(f)["results"]

with open("data/paroutes/n1-routes.json") as f:
    ref = json.load(f)

def get_leaves(node):
    if node.get("type") == "mol" and node.get("in_stock", False):
        m = Chem.MolFromSmiles(node["smiles"])
        return {Chem.MolToSmiles(m)} if m else {node["smiles"]}
    l = set()
    for c in node.get("children", []):
        l.update(get_leaves(c))
    return l

refs = {}
for r in ref:
    m = Chem.MolFromSmiles(r["smiles"])
    if m:
        refs[Chem.MolToSmiles(m)] = get_leaves(r)

for r in res[:10]:
    s = r["smiles"]
    print("Mol:", s)
    if s in refs:
        print("  REF:", list(refs[s]))
    else:
        print("  REF: NOT FOUND")
    for t in ["1_dag", "2_validated", "3_smarts"]:
        if t in r["tiers"]:
            print(f"  PRED {t}:", r["tiers"][t].get("predicted_leaves"))
    print()
