import requests
import sys

uniprot = sys.argv[1]

versions = ["v4", "v3", "v2", "v1"]

for ver in versions:
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_{ver}.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        fname = f"AF-{uniprot}.pdb"
        with open(fname, "wb") as f:
            f.write(r.content)
        print("downloaded", fname, "using", ver)
        break
else:
    print("not found")
