from PIL import Image, ImageDraw
import numpy as np
import torch
from models import SinusoidalPosEmb
from diffusion import ProjectedGaussianDiffusion
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import torch.nn.functional as F

from util import to_device
class JigsawPuzzle(nn.Module):
    def __init__(self, size=128, square_size=32, circle_size=32, seed=None):
        super().__init__()
        self.size = size
        self.circle_size = circle_size
        self.rng = np.random.default_rng(seed=seed)
        self.square_pos = self.rng.integers((circle_size + square_size)//2,
                                            size - (circle_size + square_size)//2,
                                            size=2)
        self.circle_pos = self.rng.integers(-circle_size//2, circle_size//2, size=2) + self.square_pos

        self.register_buffer("x_0", (torch.from_numpy(self.circle_pos)-self.size/2) * 8.0/self.size)
        self.square_coords = np.array([self.square_pos - square_size//2, self.square_pos + square_size//2])
        self.circle_coords = np.array([self.circle_pos - circle_size//2, self.circle_pos + circle_size//2])

    def draw_true(self):
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        draw.ellipse(list(self.circle_coords.ravel()), fill="blue")
        return image

    def _draw(self, circ_pos):
        # We treat the image as being 8 standard deviations wide
        pixel_pos = np.round((self.size * circ_pos / 8) + self.size / 2).numpy()
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        # Take chunk out to show true position
        draw.ellipse(list(self.circle_coords.ravel()), fill="white")
        offset_circ_coords = np.array([pixel_pos - self.circle_size // 2, pixel_pos + self.circle_size // 2])
        draw.ellipse(list(offset_circ_coords.ravel()), fill="blue")
        return to_tensor(image)

    def forward(self, circ_positions: torch.Tensor):
        posshape = circ_positions.shape
        if posshape == (2,):
            return self._draw(circ_positions)
        elif posshape[-1] == 2:
            flatpos = circ_positions.reshape(-1, 2)
            images = torch.stack([self._draw(x) for x in flatpos], dim=0)
            return images.reshape(*posshape[:-1], *images.shape[-3:]).to(self.x_0.device)




class CoordConv(nn.Module):
    def __init__(self, size=128, dim=16):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim)
        self.net = nn.Sequential(
            nn.Conv2d(5+dim, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 2, 3, 1, 1),
            )
        lin = torch.linspace(-1,1, steps=size)

        coords = torch.stack(torch.meshgrid(lin, lin), dim=0)[None, ...]
        self.register_buffer("coords", coords) # Register as self.coords and make sure tensors get passed to GPU

    def forward(self, x, t):
        t_emb = self.emb(t)
        batchsize = x.shape[0]
        exp_coords = self.coords.expand(batchsize, -1, -1, -1)
        exp_t_emb = t_emb[...,None,None].expand(-1,-1, *x.shape[-2:])
        nn_in = torch.cat((x, exp_coords, exp_t_emb), dim=1)
        return self.net(nn_in)

# Quick and dirty convolutional network
convnet = CoordConv()

STEPS=1000
BATCH=256
if __name__ =="__main__":
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    convnet = convnet.to(device)
    convnet.train()
    optim = torch.optim.Adam(convnet.parameters(), lr=3e-4)
    diffusion = ProjectedGaussianDiffusion(convnet).to(device)
    for i in range(40000):
        jp = JigsawPuzzle().to(device)
        truepos = jp.x_0
        loss = diffusion(truepos.repeat(BATCH,1), jp)
        print(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save(convnet.state_dict(), "weights_jig-trans.pt")