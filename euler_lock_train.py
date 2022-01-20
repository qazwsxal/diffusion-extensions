import numpy as np
import torch
import torch.nn as nn

from diffusion import GaussianDiffusion
from models import SinusoidalPosEmb, ResLayer
from math import pi
from util import *


class EulerRotPredict(nn.Module):
    def __init__(self, d_model=255):
        super().__init__()
        in_channels = 3
        t_emb_dim = d_model - in_channels

        self.time_embedding = SinusoidalPosEmb(t_emb_dim)
        self.net = nn.Sequential(
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            ResLayer(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU())),
            nn.Linear(d_model, 3),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_emb = self.time_embedding(t)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        xt = torch.cat((x, t_emb), dim=-1)

        out = self.net(xt)
        return out


BATCH = 32

if __name__ == "__main__":
    import wandb

    wandb.init(project='SO3EulerDiffusion', entity='qazwsxal', config={"diff_type": "euler"})

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = EulerRotPredict().to(device)
    net.train()
    wandb.watch(net)
    process = GaussianDiffusion(net, loss_type="l2", image_size=None).to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)

    R_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(pi / 3), torch.tensor(0.0))[None].to(device)
    R_2 = euler_to_rmat(torch.tensor(0.0), torch.tensor(2 * pi / 3), torch.tensor(0.0))[None].to(device)
    sumloss = 0
    for i in range(100000):
        weight = torch.rand(BATCH, 1).to(device)
        rmats = so3_lerp(R_1, R_2, weight)
        truepos = torch.stack(rmat_to_euler(rmats), dim=-1)
        loss = process(truepos)
        optim.zero_grad()
        loss.backward()
        optim.step()
        sumloss += loss.detach().cpu().item()
        if i % 10 == 0:
            wandb.log({"loss": sumloss / 10})
            sumloss = 0
        if i % 1000 == 0:
            torch.save(net.state_dict(), "weights/weights_euler_lock.pt")
    torch.save(net.state_dict(), "weights/weights_euler_lock.pt")