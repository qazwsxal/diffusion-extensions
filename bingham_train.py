import torch.nn as nn

from diffusion import SO3Diffusion
from distributions import Bingham
from models import SinusoidalPosEmb
from util import *


class RotPredict(nn.Module):
    def __init__(self, d_model=65, out_type="rotmat", in_type="rotmat"):
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
        xt = torch.cat((x_flat, t_emb), dim=-1)

        out = self.net(xt)
        if self.out_type == "rotmat":
            out = six2rmat(out)
        return out


BATCH = 64
# device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

# Small, uncorrelated rotations
cov1 = torch.diag(torch.tensor([1000.0, 0.1, 0.1, 0.1], device=device))
# Small, similar-axis rotations, ijk parts need to be correlated
cov2 = torch.tensor([
    [1e05, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.99, 0.99],
    [0.00, 0.99, 1.00, 0.99],
    [0.00, 0.99, 0.99, 1.00],
], device=device)
#
# Big, similar-axis rotations, ijk parts need to be correlated
cov3 = torch.tensor([
    [1.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.90, 0.90],
    [0.00, 0.90, 1.00, 0.90],
    [0.00, 0.90, 0.90, 1.00],
], device=device)
# unit gaussian
cov4 = torch.eye(4, device=device)
loc = torch.zeros(4, device=device)
covpairs = (("Small Uncorrelated Rotations", "sur", cov1),
            ("Small Correlated Rotations", "scr", cov2),
            ("Large Correlated Rotations", "lcr", cov3),
            ("Large Uncorrelated Rotations", "lur", cov4),
            )

if __name__ == "__main__":
    import tqdm

    for title, acro, cov in covpairs:
        net = RotPredict(out_type="skewvec").to(device)
        net.train()
        process = SO3Diffusion(net, loss_type="skewvec").to(device)
        optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)
        dist = Bingham(loc, covariance_matrix=cov)
        for i in tqdm.trange(100000):
            truepos = quat_to_rmat(dist.sample((BATCH,)))
            loss = process(truepos)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                print(loss.item())
            if i % 1000 == 0:
                torch.save(net.state_dict(), f"weights/weights_bing_{acro}_{i}.pt")
