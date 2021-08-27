import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import ShapeNet
from diffusion import ProjectedSO3Diffusion, extract
from distributions import IsotropicGaussianSO3
from models import SinusoidalPosEmb, Siren, ResLayer, PointCloudProj
from rotations import skew2vec, log_rmat


class RotPredict(nn.Module):
    def __init__(self, d_model=512, nhead=8, layers=12, out_type="backprop"):
        super().__init__()
        self.out_type = out_type
        if out_type in ["skewvec", "backprop"]:
            d_out = 3
        else:
            RuntimeError(f"Unexpected out_type: {out_type}")
        self.recip_sqrt_dim = 1 / np.sqrt(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.position_siren = Siren(in_channels=3, out_channels=d_model // 2, scale=30)
        self.time_embedding = SinusoidalPosEmb(d_model // 2)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        if self.out_type == "skewvec":
            self.out_query = nn.Linear(d_model // 2, d_model)
            self.out_key = nn.Linear(d_model, d_model)
            self.out_value = nn.Linear(d_model, d_model)

        out_net = [ResLayer(nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.SiLU(),
                                          )
                            )
                   for _ in range(4)]
        out_net.append(nn.Linear(d_model, d_out))
        self.out_net = nn.Sequential(*out_net)

    def forward(self, x, t):
        x_emb = self.position_siren(x)
        t_emb = self.time_embedding(t)
        t_in = torch.cat((x_emb, t_emb[:, None, :].expand(x_emb.shape)), dim=2)
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        encoding = self.encoder(t_in.transpose(0, 1)).transpose(0, 1)

        if self.out_type == "skewvec":
            # Manual single-token attention for final prediction
            Q = self.out_query(torch.ones_like(t_emb[:, None, :]))
            K = self.out_key(encoding)
            V = self.out_value(encoding)
            t_out = torch.softmax(Q @ K.transpose(-1, -2) * self.recip_sqrt_dim, dim=-1) @ V
            t_out = t_out[..., 0, :]  # Drop sequence/token dimension
        elif self.out_type == "backprop":
            t_out = encoding

        out = self.out_net(t_out)
        return out


BATCH = 2

if __name__ == "__main__":
    import wandb

    wandb.init(project='ProjectedSO3Diffusion', entity='qazwsxal')

    torch.autograd.set_detect_anomaly(True)
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    net = RotPredict().to(device)
    net.train()
    wandb.watch(net, log="all", log_freq=10)
    process = ProjectedSO3Diffusion(net).to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=1e-5)
    i = 0
    data = next(iter(dl)).repeat(BATCH, 1, 1)
    proj = PointCloudProj(data.to(device)).to(device)
    while i < 400000:
        truepos = process.identity
        loss = process(truepos.repeat(BATCH, 1, 1), proj)
        print(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        logdict = {"loss": loss.detach()}
        # Initial setup and prediction of some test data.
        if i == 0:
            t = torch.randint(0, process.num_timesteps, (BATCH,), device=device).long()
            eps = extract(process.sqrt_one_minus_alphas_cumprod, t, t.shape)
            noise = IsotropicGaussianSO3(eps).sample().detach()
            x_noisy = process.q_sample(x_start=truepos.repeat(BATCH, 1, 1), t=t, noise=noise)
            x_noisy.requires_grad = True
            proj_x_noisy = process.projection(x_noisy)
            descaled_noise = skew2vec(log_rmat(noise)) * (1 / eps)[..., None]

        if i % 100 == 0:
            torch.save(net.state_dict(), "weights_aircraft.pt")
            with torch.no_grad():
                x_recon = process.denoise_fn(proj_x_noisy, t)
            orth_loss = (proj_x_noisy * x_recon).sum(dim=-1).pow(2).mean()
            r_grad = torch.autograd.grad(proj_x_noisy, x_noisy, x_recon, retain_graph=True, create_graph=False)[0]
            s_v = r_grad @ x_noisy.transpose(-1, -2)
            # Extract skew-symmetric part i.e. project onto tangent
            s_v_proj = (s_v - s_v.transpose(-1, -2)) / 2
            sym_part = (s_v + s_v.transpose(-1, -2)) / 2
            sym_loss = sym_part.pow(2).mean()
            # Convert to vector form for regression
            predict = skew2vec(s_v_proj)
            test_loss = F.mse_loss(predict, descaled_noise) + sym_loss + orth_loss
            logdict["test loss"] = test_loss.detach()

            start = proj_x_noisy[0].detach().cpu().numpy()[::16] / 2
            end = (proj_x_noisy[0] - x_recon[0]).detach().cpu().numpy()[::16] / 2

            logdict["Predicted Gradients"] = wandb.Object3D(
                {"type": "lidar/beta",
                 "points": proj_x_noisy[0].detach().cpu().numpy() / 2,
                 "vectors": np.array([{"start": s.tolist(), "end": e.tolist()}
                                      for s, e in zip(start, end)
                                      ]),
                 "boxes": np.array([{
                     "corners": [
                         [-0.5, -0.5, -0.5],
                         [-0.5, 0.5, -0.5],
                         [-0.5, -0.5, 0.5],
                         [0.5, -0.5, -0.5],
                         [0.5, 0.5, -0.5],
                         [-0.5, 0.5, 0.5],
                         [0.5, -0.5, 0.5],
                         [0.5, 0.5, 0.5]
                         ],
                     # "label": "Tree",
                     "color": [123, 321, 111],
                     }, ])
                 }
                )
        i += 1
        wandb.log(logdict)

    torch.save(net.state_dict(), "weights_aircraft.pt")