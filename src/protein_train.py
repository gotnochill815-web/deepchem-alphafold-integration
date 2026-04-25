from Bio.PDB import PDBParser, PDBList
import numpy as np
import deepchem as dc
import os

ids = ["1CRN", "1UBQ", "1VII", "1L2Y"]
labels = []

pdbl = PDBList()
parser = PDBParser(QUIET=True)

X = []

for pid in ids:
    pdbl.retrieve_pdb_file(pid, pdir=".", file_format="pdb")
    fname = f"pdb{pid.lower()}.ent"

    structure = parser.get_structure(pid, fname)

    coords = []
    for residue in structure.get_residues():
        if "CA" in residue:
            coords.append(residue["CA"].coord)

    coords = np.array(coords)
    n = len(coords)

    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(coords[i]-coords[j]))

    feats = [
        n,
        np.mean(dists) if dists else 0,
        np.std(dists) if dists else 0,
        np.max(dists) if dists else 0
    ]

    X.append(feats)
    labels.append([1 if n > 60 else 0])

X = np.array(X, dtype=float)
y = np.array(labels)

dataset = dc.data.NumpyDataset(X, y)

model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=4,
    layer_sizes=[32,32]
)

model.fit(dataset, nb_epoch=25)

print("training complete")
print(X)
print(y)
