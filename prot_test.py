from torch.utils.data import DataLoader

from models import ProtNet
from prot_util import *
from util import identity, to_device, init_from_dict
from diffusion import ProjectedSE3Diffusion, ProjectedEulerDiffusion
from itertools import count
from tqdm import tqdm, trange
import pickle

AUGMENT = True
SAMPLES = 4

if __name__ == "__main__":
    import os
    # Windows doesn't support process forking
    if  os.name != 'nt':
        torch.multiprocessing.set_start_method("forkserver")
    import wandb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="batch",
        )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
        )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="transformer dimension",
        )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="number of self-attention heads per layer",
        )
    parser.add_argument(
        "--dim_head",
        type=int,
        default=32,
        help="dimension of self-attention head",
        )
    parser.add_argument(
        "--t_depth",
        type=int,
        default=12,
        help="number of transformer layers",
        )
    parser.add_argument(
        "--c_depth",
        type=int,
        default=8,
        help="number of residue convolutional layers",
        )
    parser.add_argument(
        "--se3",
        action='store_true',
        help="Use SE3 diffusion rather than euler angles",
        )
    args = parser.parse_args()

    config = vars(args)


    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = ProtDataset("data/BPTI_dock")
    dl = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                    num_workers=0, pin_memory=True,
                    collate_fn=identity,
                    # persistent_workers=True,
                    )

    net, = init_from_dict(config, ProtNet)
    net.to(device)
    diff_type = "se3" if config['se3'] else "eul"
    weight_path = f"weights/weights_protein_{diff_type}.pt"
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()

    if config['se3']:
        process = ProjectedSE3Diffusion(net).to(device)
        true_rot = torch.eye(3).unsqueeze(0).expand(args.batch, -1, -1).to(device)
        true_shift = torch.zeros(args.batch, 3).to(device)
        true_pos = AffineT(shift=true_shift, rot=true_rot)
    else:
        process = ProjectedEulerDiffusion(net).to(device)
        true_pos = torch.zeros(args.batch, 6).to(device)


    results = []
    for i, data in enumerate(tqdm(dl, desc='batch')):
        data = to_device(device, *data)
        # Random transform.
        if AUGMENT:
            with torch.no_grad():
                transl = torch.randn((len(data), 3)).to(device)
                rot = torch.linalg.qr(torch.randn((len(data), 3, 3)))[0].to(device)
                aff_ts = [AffineT(shift=t, rot=r) for t,r in zip(transl, rot)]
                data = [move_prots(t, p)for t,p in zip(aff_ts, data)]

        projection = ProtProjection(data, se3=config['se3']).to(device)

        process.projection = projection

        samples = []

        for samp in trange(SAMPLES, leave=False, desc="sample number"):
            with torch.no_grad():
                # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
                R, _ = torch.linalg.qr(torch.randn((args.batch, 3, 3)), "reduced")
                T = torch.randn((args.batch, 3))
                if not config['se3']:
                    R = torch.stack(rmat_to_euler(R),dim=-1)
                    transform = torch.cat((R,T), dim=-1)
                else:
                    transform = AffineT(rot=R, shift=T)
                transform = transform.to(device)

                for i in tqdm(reversed(range(0, process.num_timesteps)),
                              desc='sampling loop time step',
                              total=process.num_timesteps,
                              leave=False,
                              ):
                    transform = process.p_sample(transform, torch.full((args.batch,), i, device=device,
                                                             dtype=torch.long)).detach()
            # TODO make this SE3 compatible
            if not config['se3']:

                eul = transform[..., :3]
                rots = euler_to_rmat(*torch.unbind(eul,-1))
                shift = transform[..., 3:]
                aff_t = AffineT(rots, shift).to('cpu')
            else:
                aff_t = transform.to('cpu')
            samples.append(aff_t)
        results.append(samples)
pickle.dump(results, open(f'prot_samples_{diff_type}.pkl', 'wb'))