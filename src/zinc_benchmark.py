import pandas as pd
import requests
import numpy as np
import deepchem as dc
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("zinc_dataset.csv")
parser = PDBParser(QUIET=True)

mapping = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
    "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
    "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

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

def parse(fname):
    structure = parser.get_structure("p", fname)
    seq = []
    conf = []

    for residue in structure.get_residues():
        name = residue.get_resname()
        if name in mapping:
            seq.append(mapping[name])
            if "CA" in residue:
                conf.append(residue["CA"].bfactor)

    return "".join(seq), np.array(conf)

def motif_count(seq):
    count = 0
    for i in range(len(seq)-5):
        chunk = seq[i:i+6]
        if chunk.count("C") >= 2 and chunk.count("H") >= 1:
            count += 1
    return count

X, y = [], []

for _, row in df.iterrows():
    uid = row["uniprot"]
    label = row["label"]

    print("processing", uid)
    file = download_af(uid)
    if file is None:
        continue

    try:
        seq, conf = parse(file)

        L = max(len(seq),1)

        feats = [
            len(seq),
            seq.count("C")/L,
            seq.count("H")/L,
            seq.count("C")+seq.count("H"),
            motif_count(seq),
            np.mean(conf) if len(conf) else 0,
            np.mean(conf < 70) if len(conf) else 0
        ]

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

model.fit(train, nb_epoch=80)

pred = model.predict(test)[:,0,1]
labels = (pred > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, labels))
print("F1:", f1_score(y_test, labels))
print("done")
