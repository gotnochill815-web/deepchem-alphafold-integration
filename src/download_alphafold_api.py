import requests
import sys

uniprot = sys.argv[1]

# Try public API endpoint
url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

r = requests.get(url)

if r.status_code != 200:
    print("API failed:", r.status_code)
    exit()

data = r.json()

if not data:
    print("No entry found")
    exit()

entry = data[0]

# Common fields used historically
pdb_url = entry.get("pdbUrl") or entry.get("pdb_url")
cif_url = entry.get("cifUrl") or entry.get("cif_url")

download_url = pdb_url if pdb_url else cif_url

if not download_url:
    print("No downloadable structure URL in API response")
    print(entry.keys())
    exit()

ext = ".pdb" if pdb_url else ".cif"
fname = f"AF-{uniprot}{ext}"

file_data = requests.get(download_url)

with open(fname, "wb") as f:
    f.write(file_data.content)

print("downloaded", fname)
print("source:", download_url)
