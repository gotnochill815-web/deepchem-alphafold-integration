from Bio.PDB import PDBParser
import numpy as np
import sys

file = sys.argv[1]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", file)

coords = []
conf = []

for residue in structure.get_residues():
    if "CA" in residue:
        atom = residue["CA"]
        coords.append(atom.coord)
        conf.append(atom.bfactor)   # AlphaFold pLDDT stored here

coords = np.array(coords)
conf = np.array(conf)

n = len(coords)

# Pairwise distances
dists = []
for i in range(n):
    for j in range(i + 1, n):
        d = np.linalg.norm(coords[i] - coords[j])
        dists.append(d)

dists = np.array(dists) if len(dists) else np.array([0.0])

features = [
    n,                              # length
    float(np.mean(dists)),          # mean distance
    float(np.std(dists)),           # distance spread
    float(np.max(dists)),           # max span
    float(np.mean(conf)),           # mean confidence
    float(np.std(conf)),            # confidence spread
    float(np.sum(conf < 70) / n) if n > 0 else 0.0   # low-confidence fraction
]

print("ALPHAFOLD_FEATURES:", features)
