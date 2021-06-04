from PIL import Image, ImageDraw
import numpy as np
import torch
from diffusion import GaussianDiffusionProcess
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import torch.nn.functional as F

from util import to_device
class JigsawPuzzle(object):
    def __init__(self, size=128, square_size = 32, circle_size = 32, seed=None):
        self.size = size
        self.circle_size = circle_size
        self.rng = np.random.default_rng(seed=seed)
        self.square_pos = self.rng.integers((circle_size + square_size)//2,
                                            size - (circle_size + square_size)//2,
                                            size=2)
        self.circle_pos = self.rng.integers(-circle_size//2, circle_size//2, size=2) + self.square_pos

        self.x_0 = (torch.from_numpy(self.circle_pos)-self.size/2) * 6.0/self.size
        self.square_coords = np.array([self.square_pos - square_size//2, self.square_pos + square_size//2])
        self.circle_coords = np.array([self.circle_pos - circle_size//2, self.circle_pos + circle_size//2])

    def draw_true(self):
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        draw.ellipse(list(self.circle_coords.ravel()), fill="blue")
        return image
    def draw_diffuse(self, circ_pos):
        # We treat the image as being 6 standard deviations wide
        # This means that a circle at the center of the image
        # must travel 3 standard deviations (~0.27% chance)
        # To have its center on the edge of the image.
        pixel_pos = np.round((self.size * circ_pos / 6)+self.size/2).numpy()
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        # Take chunk out to show true position
        draw.ellipse(list(self.circle_coords.ravel()), fill="white")
        offset_circ_coords = np.array([pixel_pos - self.circle_size//2, pixel_pos + self.circle_size//2])
        draw.ellipse(list(offset_circ_coords.ravel()), fill="blue")
        return image


class JigsawGenerator(IterableDataset):
    def __init__(self, size=128, square_size = 32, circle_size = 32, steps=500, schedule='linear'):
        self.size = size
        self.square_size = square_size
        self.circle_size = circle_size
        self.steps = steps
        self.process = GaussianDiffusionProcess(steps=steps, schedule=schedule)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = torch.seed() # set seed for torch to non-deteministic value and use it for JigsawPuzzle
        else:
            seed = None

        while True:
            step = torch.randint(0,self.steps, (1,))
            t = step/self.steps
            jp = JigsawPuzzle(size=self.size, square_size=self.square_size, circle_size=self.circle_size, seed=seed)
            offset, eps = self.process.x_t(jp.x_0, step)
            image = jp.draw_diffuse(offset)
            yield to_tensor(image), eps, t


# Quick and dirty convolutional network
convnet = nn.Sequential(
    nn.Conv2d(4, 32, 3, 1, 1),
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

if __name__ =="__main__":
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    gen = JigsawGenerator()
    dl = DataLoader(gen, batch_size=256, pin_memory=True, num_workers=4)
    convnet = convnet.to(device)
    convnet.train()
    optim = torch.optim.Adam(convnet.parameters(), lr=3e-4)
    for i, (data) in enumerate(dl):
        imgs, eps, t = to_device(device, *data, non_blocking=True)
        # Concat time information
        nn_in = torch.cat((imgs, t[...,None,None].expand(-1, -1, gen.size, gen.size)), dim=1)
        out = convnet(nn_in).mean(dim=(-1,-2))
        loss = F.mse_loss(out, eps)
        print(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i == 4000: # ~5 mins training
            break
    torch.save(convnet.state_dict(), "weights_jig-trans.pt")