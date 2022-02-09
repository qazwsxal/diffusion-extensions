import os
from pathlib import Path
from typing import List, Union

import Bio.PDB as PDB
from torch import nn
from torch.utils.data import Dataset

from util import *

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

RES_COUNT = len(UNIQUE_RESIDUES)


def pdb_2_rigid_gas(pdbfile) -> ProtData:
    structure = PDB.PDBParser().get_structure("null", pdbfile)
    residues = list(structure.get_residues())
    res_one_hot = torch.zeros((len(residues), len(UNIQUE_RESIDUES)))
    res_pos = torch.zeros((len(residues), 3))
    res_vecs = torch.zeros((len(residues), 3, 3))
    for i, res in enumerate(residues):
        res_one_hot[i, UNIQUE_RESIDUES.index(res.resname)] = 1
        res_pos[i, :] = torch.from_numpy(res['CA'].coord) # Points are spread out, normalise to smaller values
        C_CA = torch.from_numpy(res['C'].coord - res['CA'].coord)
        N_CA = torch.from_numpy(res['N'].coord - res['CA'].coord)
        v1 = C_CA / C_CA.norm()
        v2 = N_CA / N_CA.norm()
        v3 = torch.cross(v1, v2)
        res_vecs[i] = torch.stack((v1, v2, v3), dim=0)
    return ProtData(res_one_hot, res_pos, res_vecs)


def move_prots(transf: AffineT, proteins: Iterable[ProtData]) -> List[ProtData]:
    """Move a collection of proteins based on a shared middle to rotate around
    """
    rot_m = transf.rot
    shift = transf.shift
    all_pos = torch.cat([x.positions for x in proteins], dim=0)
    mean_pos = all_pos.mean(dim=-2, keepdim=True)
    p_pos = [((x.positions - mean_pos) @ rot_m.transpose(-1, -2)) + mean_pos + shift for x in proteins]
    p_angs = [x.angles @ rot_m.transpose(-1, -2) for x in proteins]
    return [ProtData(x.residues, p, a) for x, p, a in zip(proteins, p_pos, p_angs)]


def move_prot(transf: AffineT, protein: ProtData) -> ProtData:
    """Move a single protein
    """
    rot_m = transf.rot
    shift = transf.shift
    mean_pos = protein.positions.mean(dim=-2, keepdim=True)
    l_pos = ((protein.positions - mean_pos) @ rot_m.transpose(-1, -2)) + mean_pos + shift
    l_angs = protein.angles @ rot_m.transpose(-1, -2)
    return ProtData(protein.residues, l_pos, l_angs)


class ProtDataset(Dataset):
    def __init__(self, path):
        super(ProtDataset, self).__init__()
        self.basepath = Path(path)
        self.prots = list({x[:4] for x in os.listdir(path)
                           if x[-3:] == "pdb" and ("receptors" in x or "ligand" in x)})
        self.prots.sort()

    def __len__(self):
        return len(self.prots)

    def __getitem__(self, idx) -> Tuple[ProtData, ProtData]:
        receptor = pdb_2_rigid_gas(self.basepath / (self.prots[idx] + "_receptors.pdb"))
        ligand = pdb_2_rigid_gas(self.basepath / (self.prots[idx] + "_ligand.pdb"))

        return receptor, ligand


class ProtProjection(nn.Module):
    def __init__(self, data: Iterable[Tuple[ProtData, ProtData]], se3=True):
        super().__init__()
        self.data = data
        self.se3 = se3

    def forward(self, transforms: Union[AffineT, torch.Tensor]):
        if self.se3:
            tfs = transforms
        else:
            eul = transforms[..., :3]
            rots = euler_to_rmat(*torch.unbind(eul, -1))
            tfs = AffineT(rots, transforms[..., 3:])
        newligs = [move_prot(t, x[1]) for t, x in zip(tfs, self.data)]
        proj_prots = [(old[0], lig) for old, lig in zip(self.data, newligs)]
        return proj_prots


if __name__ == "__main__":
    import warnings

    data_l = pdb_2_rigid_gas("data/BPTI_dock/1BTH_ligand.pdb")
    data_r = pdb_2_rigid_gas("data/BPTI_dock/1BTH_receptors.pdb")
    a = ProtDataset("data/BPTI_dock")

    shift = torch.randn(3)
    rot, _ = torch.linalg.qr(torch.randn(3, 3))
    transf = AffineT(rot=rot, shift=shift)
    rec, lig = a[5]
    rec_aug, lig_aug = move_prots(transf, (rec, lig))

    files = [os.path.join(r, f) for r, ds, fs in os.walk("data/BPTI_dock") for f in fs if f[-3:] == "pdb"]
    files = [f for f in files if "receptors" in f or "ligand" in f]
    errors = dict()
    warnings.filterwarnings("error")
    for protf in files:
        try:
            out = pdb_2_rigid_gas(protf)
        except Exception as e:
            print(protf, e)

    projector = ProtProjection([(rec_aug, lig_aug)])
    transf2 = [AffineT(rot=torch.linalg.qr(torch.randn(3, 3))[0], shift=torch.randn(3))]
    nn_in = projector(transf2)
    dataset = ProtDataset("data/BPTI_dock")

    positions = [torch.cat((receptor.positions, ligand.positions), dim=0) for receptor, ligand in dataset]
    for i,x in enumerate(positions):
        print(dataset.prots[i], x.std())
    print('aaa')