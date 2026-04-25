import requests
import numpy as np
import deepchem as dc
from Bio.PDB import PDBParser

proteins = [
    ("P69905", 1),   # Hemoglobin alpha
    ("P68871", 1),   # Hemoglobin beta
    ("P01308", 0),   # Insulin
    ("P00698", 0),   # Lysozyme
    ("P0CG47", 0),   # Ubiquitin
]

parser = PDBParser(QUIET=True)

def download_af(uniprot):
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    r = requests.get(api)
    data = r.json()[0]
    url = data["pdbUrl"]
    fname = f"{uniprot}.pdb"
    content = requests.get(url).content
    with open(fname, "wb") as f:
        f.write(content)
    return fname

def featurize(fname):
    structure = parser.get_structure("p", fname)

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

    return [
        n,
        np.mean(dists),
        np.std(dists),
        np.max(dists),
        np.mean(conf),
        np.std(conf),
        np.sum(conf < 70)/n
    ]

X = []
y = []

for pid, label in proteins:
    print("processing", pid)
    file = download_af(pid)
    feats = featurize(file)
    X.append(feats)
    y.append([label])

X = np.array(X, dtype=float)
y = np.array(y)

dataset = dc.data.NumpyDataset(X, y)

model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=7,
    layer_sizes=[32,32]
)

model.fit(dataset, nb_epoch=40)

pred = model.predict(dataset)

print("FEATURES")
print(X)
print("LABELS")
print(y)
print("PREDICTIONS")
print(pred)
print("benchmark complete")
