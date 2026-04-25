import pandas as pd
import requests
import numpy as np
import deepchem as dc
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("dna_dataset.csv")
parser = PDBParser(QUIET=True)

def download_af(uid):
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uid}"
    r = requests.get(api)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    url = data[0]["pdbUrl"]
    fname = f"{uid}.pdb"
    with open(fname,"wb") as f:
        f.write(requests.get(url).content)
    return fname

def featurize(fname):
    structure = parser.get_structure("p", fname)
    coords, conf = [], []

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
        for j in range(i+1,n):
            dists.append(np.linalg.norm(coords[i]-coords[j]))

    dists = np.array(dists) if len(dists) else np.array([0.0])

    return [
        n,
        np.mean(dists),
        np.std(dists),
        np.max(dists),
        np.mean(conf),
        np.std(conf),
        np.mean(conf < 70)
    ]

X, y = [], []

for _, row in df.iterrows():
    uid = row["uniprot"]
    label = row["label"]

    print("processing", uid)
    file = download_af(uid)
    if file is None:
        continue

    try:
        feats = featurize(file)
        X.append(feats)
        y.append(label)
    except:
        pass

X = np.array(X, dtype=float)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

train = dc.data.NumpyDataset(X_train, y_train.reshape(-1,1))
test  = dc.data.NumpyDataset(X_test, y_test.reshape(-1,1))

model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=7,
    layer_sizes=[64,64]
)

model.fit(train, nb_epoch=50)

pred = model.predict(test)[:,0,1]
labels = (pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, labels))
print("F1:", f1_score(y_test, labels))
print("done")
