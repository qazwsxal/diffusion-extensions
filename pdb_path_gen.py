import os
import pickle
import shutil
from pathlib import Path


import Bio.PDB as PDB
import torch

from prot_util import AffineT

IN_PATH = "data/BPTI_dock"
OUT_PATH = "prot_paths"


basepath = Path(IN_PATH)
prots = list({x[:4] for x in os.listdir(IN_PATH)
              if x[-3:] == "pdb" and ("receptors" in x or "ligand" in x)})
prots.sort()
# receptor ligand filepath pairs.
protpaths = [(basepath / (p + "_receptors.pdb"), basepath / (p + "_ligand.pdb")) for p in prots]

prot_count = len(protpaths)

cpu_paths = pickle.load(open('se3_paths.pkl', 'rb'))

out_dir = Path(OUT_PATH)
io=PDB.PDBIO()


for i, (receptor, ligand) in enumerate(protpaths):
    out_rec = out_dir/receptor.parts[-1]

    lig_struct = PDB.PDBParser().get_structure("null", ligand)
    for step, tfs in enumerate(cpu_paths):
        tf = tfs[i]
        new_struct= lig_struct.copy()
        rot = tf.rot.cpu().numpy()
        shift = tf.shift.cpu().numpy() * 40
        new_struct.transform(rot, shift)
        lig_file = f'{ligand.stem}_{step:04}.pdb'
        io.set_structure(new_struct)
        io.save(str(out_dir/lig_file))
    shutil.copy2(receptor, out_rec)
