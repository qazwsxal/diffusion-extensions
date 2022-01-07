import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import ShapeNet
from diffusion import ProjectedSO3Diffusion, extract
from distributions import IsotropicGaussianSO3
from models import SinusoidalPosEmb, Siren, ResLayer, PlaneNet, PointCloudProj
from util import skew2vec, log_rmat, init_from_dict, cycle


if __name__ == "__main__":
    import wandb
    import argparse

    parser = argparse.ArgumentParser(description="Aircraft rotation args")
    parser.add_argument(
        "--batch", type=int, default=2, help="batch size"
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
        )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="number of points to feed through transformer",
        )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="transformer dimension",
        )
    parser.add_argument(
        "--heads",
        type=int,
        default=2,
        help="number of self-attention heads per layer",
        )
    parser.add_argument(
        "--dim_head",
        type=int,
        default=32,
        help="dimension of self-attention head",
        )
    parser.add_argument(
        "--num_neighbours",
        type=int,
        default=6,
        help="number of neighbours for SE3 Transformer",
        )
    parser.add_argument(
        "--t_depth",
        type=int,
        default=4,
        help="number of transformer layers",
        )
    parser.add_argument(
        "--num_degrees",
        type=int,
        default=3,
        help="Highest order of spherical harmonics to generate in transformer",
        )

    args = parser.parse_args()
    wandb.init(project='ProjectedSO3Diffusion', entity='qazwsxal', config=args)
    config = wandb.config
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,), samples=config['samples'])
    dl = DataLoader(ds, batch_size=config['batch'], shuffle=True, num_workers=0, pin_memory=True)

    net, = init_from_dict(config, PlaneNet)
    net.to(device)
    net.train()
    wandb.watch(net, log_freq=10)
    process = ProjectedSO3Diffusion(net).to(device)
    truepos = process.identity
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=config['lr'])

    # Set up data/information for validation
    with torch.no_grad():
        v_ds = ShapeNet('valid', (0,))
        v_dl = DataLoader(v_ds, batch_size=config['batch'], shuffle=False, num_workers=0, pin_memory=True)
        t_v = torch.randint(0, process.num_timesteps, (config['batch'],), device=device).long()
        eps_v = extract(process.sqrt_one_minus_alphas_cumprod, t_v, t_v.shape)
        noise_v = IsotropicGaussianSO3(eps_v).sample().detach()
        dl_iter = cycle(v_dl)
        data_v = next(dl_iter).to(device)
        proj_v = PointCloudProj(data_v).to(device)
        x_noisy_v = process.q_sample(x_start=truepos.repeat(config['batch'], 1, 1), t=t_v, noise=noise_v)
        proj_x_noisy_v = proj_v(x_noisy_v)
        descaled_noise_v = skew2vec(log_rmat(noise_v)) * (1 / eps_v)[..., None]

    i = 0
    while i < 1000000:
        for data in dl:
            proj = PointCloudProj(data.to(device)).to(device)
            loss = process(truepos.repeat(config['batch'], 1, 1), proj)
            optim.zero_grad()
            loss.backward()
            optim.step()
            logdict = {"loss": loss.detach()}

            # Validation
            if i % 10 == 0:
                with torch.no_grad():
                    data_v = next(dl_iter).to(device)
                    proj_v = PointCloudProj(data_v).to(device)
                    x_noisy_v = process.q_sample(x_start=truepos.repeat(config['batch'], 1, 1), t=t_v, noise=noise_v)
                    proj_x_noisy_v = proj_v(x_noisy_v)
                    x_recon = process.denoise_fn(proj_x_noisy_v, t_v)
                    test_loss = F.mse_loss(x_recon, descaled_noise_v)
                    logdict["test loss"] = test_loss.detach()
                    logdict["test vals"] = x_recon.detach()
                torch.save(net.state_dict(), "weights/weights_aircraft.pt")


            i += 1
            wandb.log(logdict)

    torch.save(net.state_dict(), "weights/weights_aircraft.pt")
