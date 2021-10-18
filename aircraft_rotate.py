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
from util import init_from_dict


class RotPredict(nn.Module):
    def __init__(self, d_model=512, heads=4, layers=4, out_type="mean"):
        super().__init__()
        self.out_type = out_type
        d_out = 3
        self.recip_sqrt_dim = 1 / np.sqrt(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, heads)
        self.position_siren = Siren(in_channels=3, out_channels=d_model // 2, scale=30)
        self.time_embedding = SinusoidalPosEmb(d_model // 2)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.class_token = nn.Parameter(torch.randn(1,1,d_model))

        out_net = [ResLayer(nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.SiLU(),
                                          )
                            )
                   for _ in range(4)]
        out_net.append(nn.Linear(d_model, d_out))
        self.out_net = nn.Sequential(*out_net)

    def forward(self, x, t):
        batch = x.shape[0]
        x_emb = self.position_siren(x)
        t_emb = self.time_embedding(t)
        t_in = torch.cat((x_emb, t_emb[:, None, :].expand(x_emb.shape)), dim=2)
        if self.out_type == "token":
            t_in = torch.cat((t_in, self.class_token.repeat(batch,1,1)), dim=1)
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        encoding = self.encoder(t_in.transpose(0, 1)).transpose(0, 1)

        if self.out_type == "mean":
            t_out = encoding.mean(dim=1)
        elif self.out_type == "token":
            t_out = encoding[:,-1,:]

        out = self.out_net(t_out)
        return out



if __name__ == "__main__":
    import wandb
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
    parser.add_argument(
        "--batch", type=int, default=8, help="batch size (default: 8)"
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate",
        )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="transformer dimensionality",
        )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="number of self-attention heads per layer",
        )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="number of transformer layers",
        )
    parser.add_argument(
        "--out_type",
        type=str,
        default="token",
        help= "how to construct output values"
        )

    args = parser.parse_args()
    wandb.init(project='ProjectedSO3Diffusion', entity='qazwsxal', config=args)
    config = wandb.config

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=config['batch'], shuffle=True, num_workers=4, pin_memory=True)


    net, = init_from_dict(config, RotPredict)
    net.to(device)
    net.train()
    wandb.watch(net, log="all", log_freq=10)
    process = ProjectedSO3Diffusion(net).to(device)
    truepos = process.identity
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=config['lr'])

    # Set up data/information for valdiation
    with torch.no_grad():
        v_ds = ShapeNet('valid', (0,))
        v_dl = DataLoader(v_ds, batch_size=config['batch'], shuffle=False, num_workers=4, pin_memory=True)
        t_v = torch.randint(0, process.num_timesteps, (config['batch'],), device=device).long()
        eps_v = extract(process.sqrt_one_minus_alphas_cumprod, t_v, t_v.shape)
        noise_v = IsotropicGaussianSO3(eps_v).sample().detach()
        data_v = next(iter(v_dl)).to(device)
        proj_v = PointCloudProj(data_v).to(device)
        x_noisy_v = process.q_sample(x_start=truepos.repeat(config['batch'], 1, 1), t=t_v, noise=noise_v)
        proj_x_noisy_v = proj_v(x_noisy_v)
        descaled_noise_v = skew2vec(log_rmat(noise_v)) * (1 / eps_v)[..., None]

    i = 0
    while i < 1000000:
        for data in dl:
            proj = PointCloudProj(data.to(device)).to(device)
            loss = process(truepos.repeat(config['batch'], 1, 1), proj)
            print(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            logdict = {"loss": loss.detach()}

            # Validation
            if i % 100 == 0:
                torch.save(net.state_dict(), "weights/weights_aircraft.pt")
                with torch.no_grad():
                    x_recon = process.denoise_fn(proj_x_noisy_v, t_v)
                test_loss = F.mse_loss(x_recon, descaled_noise_v)
                logdict["test loss"] = test_loss.detach()

            i += 1
            wandb.log(logdict)

    torch.save(net.state_dict(), "weights/weights_aircraft.pt")