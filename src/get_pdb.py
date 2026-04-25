from Bio.PDB import PDBList

pdbl = PDBList()
pdbl.retrieve_pdb_file("1CRN", pdir=".", file_format="pdb")
print("download complete")
