import numpy as np
import torch
import torch.nn as nn

from diffusion import SO3Diffusion
from models import SinusoidalPosEmb, ResLayer
from rotations import *
from math import pi

from util import *


class RotPredict(nn.Module):
    def __init__(self, d_model=255, out_type="rotmat", in_type="rotmat"):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        if self.in_type == "rotmat":
            in_channels = 9
            t_emb_dim = d_model - in_channels
        if self.out_type == "skewvec":
            self.d_out = 3
        elif self.out_type == "rotmat":
            self.d_out = 6
        else:
            RuntimeError(f"Unexpected out_type: {out_type}")

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
            nn.Linear(d_model, self.d_out),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_flat = torch.flatten(x, start_dim=-2)
        t_emb = self.time_embedding(t)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x_flat.shape[0], -1)
        xt = torch.cat((x_flat, t_emb), dim=-1)

        out = self.net(xt)
        if self.out_type == "rotmat":
            out = six2rmat(out)
        return out


BATCH = 32

if __name__ == "__main__":
    import wandb

    wandb.init(project='SO3EulerDiffusion', entity='qazwsxal', config={"type": "SO3"})

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = RotPredict(out_type="skewvec").to(device)
    net.train()
    wandb.watch(net)
    process = SO3Diffusion(net, loss_type="skewvec").to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)

    R_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(pi / 3), torch.tensor(0.0))[None].to(device)
    R_2 = euler_to_rmat(torch.tensor(0.0), torch.tensor(2 * pi / 3), torch.tensor(0.0))[None].to(device)
    sumloss = 0
    for i in range(100000):
        weight = torch.rand(BATCH, 1).to(device)
        truepos = so3_lerp(R_1, R_2, weight)
        loss = process(truepos)
        if torch.isnan(loss).any():
            continue
        optim.zero_grad()
        loss.backward()
        optim.step()
        sumloss += loss.detach().cpu().item()
        if i % 10 == 0:
            wandb.log({"loss": sumloss / 10})
            sumloss = 0
        if i % 1000 == 0:
            torch.save(net.state_dict(), "weights/weights_so3_lock.pt")
    torch.save(net.state_dict(), "weights/weights_so3_lock.pt")