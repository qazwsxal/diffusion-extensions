from pymol import cmd, stored
from pathlib import Path

LOCATION = Path(r"C:\Users\thepe\Documents\PhD\diffusion-extensions\prot_paths")
RENDER_OUT = Path(r"D:\Render")
def render_path(prot_prefix):
    res_col = cmd.get_color_tuple(cmd.get_color_index('gray70'))
    lig_col = cmd.get_color_tuple(cmd.get_color_index('tv_red'))
    view = cmd.get_view()
    # Clear all objects
    for obj in cmd.get_names():
        cmd.delete(obj)
    # Load residual
    res_name = f'{prot_prefix}_receptors'
    res_path = str(LOCATION/f'{res_name}.pdb')
    cmd.load(res_path)
    cmd.color("gray70", res_name)
    for step in range(1001):
        lig_name = f'{prot_prefix}_ligand_{step:04}'
        lig_path = str(LOCATION/f'{lig_name}.pdb')
        cmd.load(lig_path)
        cmd.color("tv_red", lig_name)
        cmd.set_view(view)
        cmd.ray(1600,1200)
        cmd.png(str(RENDER_OUT/f'{prot_prefix}_{step:04}.png'))
        cmd.delete(lig_name)

cmd.extend("render_path",render_path)