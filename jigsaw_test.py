import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from jigsaw_translate import *
from matplotlib.lines import Line2D

from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from colors import *

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
batch = 8

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


res = torch.stack(samplelist, dim=0)
fig, axlist = plt.subplots(nrows=2, ncols=1, sharex=True)
axlist[0].plot(torch.arange(1001), res[...,0], alpha=0.5, c=BLUE)
axlist[1].plot(torch.arange(1001), res[...,1], alpha=0.5, c=ORANGE)

# Axis labels
axlist[1].set_xlabel("Reverse process steps")
axlist[0].set_ylabel("X Position")
axlist[1].set_ylabel("Y Position")

# Legend for all three axes
# custom_lines = [Line2D([0], [0], color=BLUE, lw=2),
#                 Line2D([0], [0], color=ORANGE, lw=2)]
# axlist[0].legend(custom_lines, ['X', 'Y'])

out = torch.cat(res,dim=0).reshape((quiv_res,quiv_res,2)).transpose(0,1)
plt.imshow(jp1(torch.tensor([[99.9,99.9]])).numpy())
X = torch.linspace(0, 128, steps=quiv_res)
Y = torch.linspace(0, 128, steps=quiv_res)
plt.quiver(X, Y, -out[...,0], -out[...,1], angles='xy')
plt.show()