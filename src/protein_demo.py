import deepchem as dc
import numpy as np

seqs = [
    "MKTLLL",
    "AAAAKK",
    "GGGGGG",
    "VVVVVV",
    "MKTAAA",
    "GGGAAA"
]

y = np.array([[1],[1],[0],[0],[1],[0]])

aa = "ACDEFGHIKLMNPQRSTVWY"

X = []
for s in seqs:
    vec = [s.count(c)/len(s) for c in aa]
    X.append(vec)

X = np.array(X)

dataset = dc.data.NumpyDataset(X, y)

model = dc.models.MultitaskClassifier(
    n_tasks=1,
    n_features=20,
    layer_sizes=[32,32]
)

model.fit(dataset, nb_epoch=20)

print("protein demo complete")
