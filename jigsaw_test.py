import torch

from jigsaw_translate import *

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

convnet.load_state_dict(torch.load("weights_jig-trans.pt", map_location=device))
convnet = convnet.to(device)

# Tracing out several paths to get an idea of what a process could look like:
process = GaussianDiffusionProcess(steps=500, schedule='cos')
jp1 = JigsawPuzzle

for i in range(process.steps, 0, -1):
    pass
    # TODO