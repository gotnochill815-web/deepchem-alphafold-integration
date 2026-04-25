from Bio.PDB import PDBParser
import numpy as np
import sys

file = sys.argv[1]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", file)

coords = []
residue_count = 0

for residue in structure.get_residues():
    if "CA" in residue:
        coords.append(residue["CA"].coord)
        residue_count += 1

coords = np.array(coords)

# Pairwise distances
dists = []
for i in range(len(coords)):
    for j in range(i+1, len(coords)):
        d = np.linalg.norm(coords[i] - coords[j])
        dists.append(d)

features = [
    residue_count,
    np.mean(dists) if dists else 0,
    np.std(dists) if dists else 0,
    np.max(dists) if dists else 0
]

print("FEATURES:", features)
