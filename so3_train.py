import numpy as np
import torch
import torch.nn as nn

from diffusion import SO3Diffusion
from models import SinusoidalPosEmb, Siren
from rotations import six2rmat

from util import *


class RotPredict(nn.Module):
    def __init__(self, d_model=65, out_type="rotmat", in_type = "rotmat"):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        if self.in_type == "rotmat":
            in_channels = 9
            t_emb_dim  = d_model - in_channels
        if self.out_type == "skewvec":
            self.d_out = 3
        elif self.out_type == "rotmat":
            self.d_out = 6
        else:
            RuntimeError(f"Unexpected out_type: {out_type}")

        self.time_embedding = SinusoidalPosEmb(t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.d_out),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_flat = torch.flatten(x, start_dim=-2)
        t_emb = self.time_embedding(t)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x_flat.shape[0], -1)
        xt = torch.cat((x_flat,t_emb), dim=-1)

        out = self.net(xt)
        if self.out_type == "rotmat":
            out = six2rmat(out)
        return out


BATCH = 64

if __name__ == "__main__":
    torch.set_anomaly_enabled(True)
    import wandb
    wandb.init(project='SO3Diffusion', entity='qazwsxal')

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = RotPredict(out_type="skewvec").to(device)
    net.train()
    wandb.watch(net)
    process = SO3Diffusion(net, loss_type="skewvec").to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)
    z90 = torch.tensor([[0.0,-1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])
    rotations = torch.stack((z90, z90.T), dim=0).to(device)
    for i in range(400000):
            i += 1
            idx = torch.randint(0,2,(BATCH,), device=device)
            truepos = rotations[idx]
            loss = process(truepos)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                wandb.log({"loss": loss})
                print(loss.item())
            if i % 1000 == 0:
                torch.save(net.state_dict(), "weights/weights_so3.pt")
