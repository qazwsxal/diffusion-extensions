import torch
import matplotlib.pyplot as plt
import numpy as np
from jigsaw_translate import *

device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")

convnet.load_state_dict(torch.load("weights_jig-trans.pt", map_location=device))
convnet = convnet.to(device)

# Tracing out several paths to get an idea of what a process could look like:
process = GaussianDiffusionProcess(steps=500, schedule='linear')
jp1 = JigsawPuzzle()
x_t, eps = process.x_t(jp1.x_0, process.steps)

for i, step in enumerate(range(process.steps-1, -1, -1)):
    print(x_t, jp1.x_0)
    t = torch.tensor(step/process.steps).to(device)
    pil_image = jp1.draw_diffuse(x_t.detach())
    pil_image.save(f"images/{i:04}.png")
    img = to_tensor(pil_image).to(device)
    nn_in = torch.cat((img, t[None,None,None].expand( 1, jp1.size, jp1.size)), dim=0)[None]
    out = convnet(nn_in).mean(dim=(-1,-2))
    x_t = process.undiffuse(x_t, out, step)

quiv_res = 16
xdim = torch.linspace(-3, 3, steps=quiv_res)
ydim = torch.linspace(-3, 3, steps=quiv_res)
grid = torch.stack(torch.meshgrid((xdim,ydim)),dim=-1).reshape((-1,2)).to(device)
res = []
t = torch.tensor(1)*0.1
for pos in grid:
    pil_image = jp1.draw_diffuse(pos)
    img = to_tensor(pil_image).to(device)
    nn_in = torch.cat((img, t[None,None,None].expand( 1, jp1.size, jp1.size)), dim=0)[None]
    res.append(convnet(nn_in).mean(dim=(-1,-2)).detach())

out = torch.cat(res,dim=0).reshape((quiv_res,quiv_res,2)).transpose(0,1)
plt.imshow(np.asarray(jp1.draw_true()))
X = torch.linspace(0, 128, steps=quiv_res)
Y = torch.linspace(0, 128, steps=quiv_res)
plt.quiver(X, Y, -out[...,0], -out[...,1], angles='xy')
plt.savefig('aaa.png')