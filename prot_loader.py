import Bio.PDB as PDB
import numpy as np
import torch
import os
from functools import lru_cache
from collections import defaultdict

UNIQUE_RESIDUES = ["ALA",
                   "CYS",
                   "ASP",
                   "GLU",
                   "PHE",
                   "GLY",
                   "HIS",
                   "ILE",
                   "LYS",
                   "LEU",
                   "MET",
                   "ASN",
                   "PRO",
                   "GLN",
                   "ARG",
                   "SER",
                   "THR",
                   "VAL",
                   "TRP",
                   "TYR",
                   # "PCA",  # What is this?
                   # "MSE",  # What is this?
                   # "AKR",  # What is this?
                   # "KCX",  # Ambiguous
                   # "GLX",  # Ambiguous
                   # "ASX",  # Ambiguous
                   "---",  # Unknown residual, padding for arrays
                   ]


def pdb_2_rigid_gas(pdbfile):
    structure = PDB.PDBParser().get_structure("null", pdbfile)
    residues = list(structure.get_residues())
    res_one_hot = torch.zeros((len(residues), len(UNIQUE_RESIDUES)))
    res_coords = torch.zeros((len(residues), 3))
    res_rotation = torch.zeros((len(residues), 3, 3))
    for i, res in enumerate(residues):
        res_one_hot[i, UNIQUE_RESIDUES.index(res.resname)] = 1
        res_coords[i] = torch.from_numpy(res['CA'].coord)
        C_CA = torch.from_numpy(res['C'].coord - res['CA'].coord)
        N_CA = torch.from_numpy(res['N'].coord - res['CA'].coord)
        v1 = C_CA
        v2 = torch.cross(C_CA, N_CA)
        v3 = torch.cross(C_CA, v2)
        # TODO:  Do I want columns or rows here?
        res_rotation[i] = torch.stack((v1, v2, v3), dim=0)
    return res_one_hot, res_coords, res_rotation


if __name__ == "__main__":
    data_l = pdb_2_rigid_gas("data/testPDBs/1A2K/1A2K_l_b.pdb")
    data_r = pdb_2_rigid_gas("data/testPDBs/1A2K/1A2K_r_b.pdb")