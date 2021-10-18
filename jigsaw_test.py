import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from jigsaw_translate import *
from matplotlib.collections import LineCollection
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_pil_image

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cpu")

convnet.load_state_dict(torch.load("weights/weights_jig-trans.pt", map_location=device))
convnet = convnet.to(device)

# Tracing out several paths to get an idea of what a process could look like:
process = ProjectedGaussianDiffusion(convnet, timesteps=STEPS)
jp1 = JigsawPuzzle(seed=1234)

samplelist = []
process.projection = jp1
device = process.betas.device
batch = 256

samples = torch.randn((batch,2), device=device).detach()
samplelist.append(samples)
fig = plt.figure()
for i in tqdm(reversed(range(0, process.num_timesteps)), desc='sampling loop time step', total=process.num_timesteps):
    samples = process.p_sample(samples, torch.full((batch,), i, device=device, dtype=torch.long)).detach()
    samplelist.append(samples)

# Render with blue circle far offscreen to show
im_clean = to_pil_image(jp1(torch.tensor([[99.9,99.9]]))[0])

for i, samples in tqdm(reversed(list(enumerate(samplelist))), total=process.num_timesteps):
    im = to_pil_image(jp1(samples).mean(dim=0))
    im.save(f"images/{i:04}.png")
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(f"Step {i}")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal', adjustable='box')
    ax.imshow(im_clean, extent=[-4,4,-4,4])
    ax.scatter(samples[:,0].cpu(), -samples[:,1].cpu(), s=1.5)
    fig.savefig(f"images/diff_{i:04}.eps", bbox_inches="tight")
    fig.clear()

# Repeat, saving intermediate positions,
# no drawing, just plotting paths
paths = []
for _ in range(3):
    x_t = x_T
    paths.append([])
    for i, step in enumerate(range(process.steps-1, -1, -1)):
        paths[-1].append(x_t.detach())
        t = torch.tensor(step/process.steps).to(device)
        pil_image = jp1(x_t.detach())
        img = to_tensor(pil_image).to(device)
        nn_in = torch.cat((img, t[None,None,None].expand( 1, jp1.size, jp1.size)), dim=0)[None]
        out = convnet(nn_in).mean(dim=(-1,-2))
        x_t = process.undiffuse(x_t, out, step)
data = (torch.stack([torch.cat(x, dim=0) for x in paths], dim=0)/6)*128+64
plt.imshow(np.asarray(jp1(x_T)))
plt.plot(data[...,0].T, data[...,1].T, alpha=0.2)


quiv_res = 16
xdim = torch.linspace(-3, 3, steps=quiv_res)
ydim = torch.linspace(-3, 3, steps=quiv_res)
grid = torch.stack(torch.meshgrid((xdim,ydim)),dim=-1).reshape((-1,2)).to(device)
res = []
t = torch.tensor(1)*0.1
for pos in grid:
    pil_image = jp1(pos)
    img = to_tensor(pil_image).to(device)
    nn_in = torch.cat((img, t[None,None,None].expand( 1, jp1.size, jp1.size)), dim=0)[None]
    res.append(convnet(nn_in).mean(dim=(-1,-2)).detach())

out = torch.cat(res,dim=0).reshape((quiv_res,quiv_res,2)).transpose(0,1)
plt.imshow(np.asarray(jp1.draw_true()))
X = torch.linspace(0, 128, steps=quiv_res)
Y = torch.linspace(0, 128, steps=quiv_res)
plt.quiver(X, Y, -out[...,0], -out[...,1], angles='xy')
plt.show()