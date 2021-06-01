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
        self.rng = np.random.default_rng(seed=seed)
        self.square_pos = self.rng.integers((circle_size + square_size)//2,
                                            size - (circle_size + square_size)//2,
                                            size=2)
        self.circle_pos = self.rng.integers(-circle_size//2, circle_size//2, size=2) + self.square_pos
        self.square_coords = np.array([self.square_pos - square_size//2, self.square_pos + square_size//2])
        self.circle_coords = np.array([self.circle_pos - circle_size//2, self.circle_pos + circle_size//2])

    def draw_true(self):
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        draw.ellipse(list(self.circle_coords.ravel()), fill="blue")
        return image
    def draw_offset(self, offset):
        # We treat the image as being 6 standard deviations wide
        # This means that a circle at the center of the image
        # must travel 3 standard deviations (~0.27% chance)
        # To have its center on the edge of the image.
        pixel_offset = np.round(self.size * offset/6).numpy()
        image = Image.new('RGB', (self.size, self.size), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(list(self.square_coords.ravel()), fill="red")
        # Take chunk out to show true position
        draw.ellipse(list(self.circle_coords.ravel()), fill="white")
        offset_circ_coords = self.circle_coords + pixel_offset[None, :]
        draw.ellipse(list(offset_circ_coords.ravel()), fill="blue")
        return image

class JigsawGenerator(IterableDataset):
    def __init__(self, size=128, square_size = 32, circle_size = 32, steps=500, schedule='cos'):
        self.size = size
        self.square_size = square_size
        self.circle_size = circle_size
        self.steps = steps
        self.process = GaussianDiffusionProcess(steps=500, schedule='cos')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = worker_info.id
            torch.seed()
        else:
            seed = None

        zeros = torch.zeros(2)
        while True:
            step = torch.randint(0,self.steps, (1,))
            offset, eps = self.process.x_t(zeros, step)
            jp = JigsawPuzzle(size=self.size, square_size=self.square_size, circle_size=self.circle_size, seed=seed)
            image = jp.draw_offset(offset)
            yield to_tensor(image), eps



if __name__ =="__main__":
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    gen = JigsawGenerator()
    dl = DataLoader(gen, batch_size=128, pin_memory=True, num_workers=4)
    # Quick and dirty convolutional network
    convnet = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 2, 3, 1, 1),
        ).to(device)

    optim = torch.optim.Adam(convnet.parameters(), lr=3e-4)
    for i, (data) in enumerate(dl):
        imgs, eps = to_device(device, *data, non_blocking=True)
        out = convnet(imgs).mean(dim=(-1,-2))
        loss = F.mse_loss(out, eps)
        print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()