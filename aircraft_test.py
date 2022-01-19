from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from datasets import ShapeNet
from diffusion import ProjectedSO3Diffusion, ProjectedGaussianDiffusion
from models import PointCloudProj, PlaneNet
from util import *

SAMPLES = 8

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aircraft test args")
    parser.add_argument(
        "--batch", type=int, default=2, help="batch size"
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
        default=512,
        help="transformer dimension",
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
        "--so3",
        action='store_true',
        help="Use SO3 diffusion rather than euler angles",
        )

    args = parser.parse_args()

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('test', (0,))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    res = torch.zeros((len(ds), SAMPLES))

    config = vars(args)
    net, = init_from_dict(config, PlaneNet)
    net.to(device)
    diff_type = "so3" if config['so3'] else "eul"
    weight_path = f"weights/weights_aircraft_{diff_type}.pt"
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    if config['so3']:
        process = ProjectedSO3Diffusion(net).to(device)
        truepos = process.identity
        truepos_repeat = truepos.repeat(config['batch'], 1, 1)
    else:
        process = ProjectedGaussianDiffusion(net).to(device)
        truepos = torch.zeros(3).to(device)
        truepos_repeat = truepos.repeat(config['batch'], 1)

    for b, data in enumerate(tqdm(dl, desc='batch')):
        proj = PointCloudProj(data.to(device), so3=config['so3']).to(device)
        process.projection = proj

        results = torch.zeros((args.batch, SAMPLES, 3, 3)).to(device)

        for samp in trange(SAMPLES, leave=False, desc="sample number"):
            with torch.no_grad():
                # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
                R, _ = torch.linalg.qr(torch.randn((args.batch, 3, 3)), "reduced")
                if not config['so3']:
                    R = torch.stack(rmat_to_euler(R),dim=-1)
                R = R.to(device)
                for i in tqdm(reversed(range(0, process.num_timesteps)),
                              desc='sampling loop time step',
                              total=process.num_timesteps,
                              leave=False,
                              ):
                    R = process.p_sample(R, torch.full((args.batch,), i, device=device, dtype=torch.long)).detach()
            if not config['so3']:
                results[:, samp] = euler_to_rmat(*torch.unbind(R,-1))
            else:
                results[:, samp] = R

        axis, angle = rmat_to_aa(results)
        start = b * args.batch
        end = start + len(angle)
        res[start:end] = angle.detach().cpu().squeeze()
    torch.save(res, f"weights/results_aircraft_{diff_type}.pt")