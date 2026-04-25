from Bio.PDB import PDBParser, PDBList
import numpy as np
import deepchem as dc

ids = ["1CRN", "1UBQ", "1VII", "1L2Y"]

pdbl = PDBList()
parser = PDBParser(QUIET=True)

X = []
y = []

for pid in ids:
    pdbl.retrieve_pdb_file(pid, pdir=".", file_format="pdb")
    fname = f"pdb{pid.lower()}.ent"

    structure = parser.get_structure(pid, fname)

    coords = []
    conf = []

    for residue in structure.get_residues():
        if "CA" in residue:
            atom = residue["CA"]
            coords.append(atom.coord)
            conf.append(atom.bfactor)

    coords = np.array(coords)
    conf = np.array(conf)

    n = len(coords)

    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(coords[i]-coords[j]))

    dists = np.array(dists) if len(dists) else np.array([0.0])

    feats = [
        n,
        np.mean(dists),
        np.std(dists),
        np.max(dists),
        np.mean(conf),
        np.std(conf),
        np.sum(conf < 70)/n
    ]

    X.append(feats)
    y.append([1 if n > 60 else 0])

X = np.array(X, dtype=float)
y = np.array(y)

dataset = dc.data.NumpyDataset(X, y)

model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=7,
    layer_sizes=[32,32]
)

model.fit(dataset, nb_epoch=30)

print("alpha pipeline complete")
print(X)
print(y)
