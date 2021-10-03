import torch

from aircraft_rotate import *
from datasets import ShapeNet
from models import PointCloudProj
from mpl_utils import *
from tqdm import tqdm, trange
from rotations import *
from diffusion import ProjectedSO3Diffusion

SAMPLES = 8

if __name__ == "__main__":
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
        help="how to construct output values"
        )

    args = parser.parse_args()

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('test', (0,))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    res = torch.zeros((len(ds), SAMPLES))

    net, = init_from_dict(vars(args), RotPredict)
    net.to(device)
    net.load_state_dict(torch.load("weights_aircraft.pt", map_location=device))
    net.eval()
    process = ProjectedSO3Diffusion(net).to(device)


    for b, data in enumerate(tqdm(dl, desc = 'batch')):
        proj = PointCloudProj(data.to(device)).to(device)
        process.projection = proj
        results = torch.zeros((args.batch, SAMPLES, 3,3)).to(device)
        for samp in trange(SAMPLES, leave=False, desc="sample number"):
            with torch.no_grad():
                # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
                R, _ = torch.linalg.qr(torch.randn((args.batch, 3, 3)), "reduced")
                R = R.to(device)
                for i in tqdm(reversed(range(0, process.num_timesteps)),
                              desc='sampling loop time step',
                              total=process.num_timesteps,
                              leave=False,
                              ):
                    R = process.p_sample(R, torch.full((args.batch,), i, device=device, dtype=torch.long)).detach()
            results[:,samp] = R

        axis, angle = rmat_to_aa(results)
        start = b * args.batch
        end = start + len(angle)
        res[start:end] = angle.detach().cpu().squeeze()
    torch.save(res, "angles.pt")

