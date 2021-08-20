import numpy as np
import torch
from aircraft_rotate import *
from mpl_utils import *
from tqdm import tqdm

from diffusion import ProjectedSO3Diffusion

BATCH = 16

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = RotPredict().to(device)
    net.load_state_dict(torch.load("weights_aircraft.pt", map_location=device))
    net.eval()
    process = ProjectedSO3Diffusion(net).to(device)

    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    iter_dl = iter(dl)
    data = next(iter_dl)
    proj = PointCloudProj(data.to(device)).to(device)
    process.projection = proj
    # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
    R, _ = torch.qr(torch.randn((BATCH, 3, 3)))
    R = R.to(device)
    res = torch.zeros((process.num_timesteps, BATCH, 3, 3), device=device)
    for i in tqdm(reversed(range(0, process.num_timesteps)),
                  desc='sampling loop time step',
                  total=process.num_timesteps,
                  ):
        res[i] = R.detach()
        R = process.p_sample(R, torch.full((BATCH,), i, device=device, dtype=torch.long)).detach()

    # Decompose into euler-angle form for plotting.
    sy = torch.sqrt(res[..., 0, 0] * res[..., 0, 0] + res[..., 1, 0] * res[..., 1, 0])
    x = torch.atan2(res[..., 2, 1], res[..., 2, 2]).detach().cpu()
    y = torch.atan2(res[..., 2, 0], sy).detach().cpu()
    z = torch.atan2(res[..., 1, 0], res[..., 0, 0]).detach().cpu()

    # Seperate X, Y, Z axis plots
    fig, axlist = plt.subplots(nrows=3, ncols=1, sharex=True)
    axlist[0].plot(torch.arange(1000).flip(0), x, alpha=0.2, c="#1f77b4")
    axlist[1].plot(torch.arange(1000).flip(0), y, alpha=0.2, c="#ff7f0e")
    axlist[2].plot(torch.arange(1000).flip(0), z, alpha=0.2, c="#2ca02c")

    # Add "target" lines to show convergence
    axlist[0].axhline(0, color='grey', linestyle="-", lw=0.5)
    axlist[1].axhline(0, color='grey', linestyle="-", lw=0.5)
    axlist[2].axhline(np.pi / 2, color='grey', linestyle="-", lw=0.5)
    axlist[2].axhline(-np.pi / 2, color='grey', linestyle="-", lw=0.5)

    # Axis labels
    axlist[2].set_xlabel("Reverse process steps")
    axlist[1].set_ylabel("Angle")

    # Legend for all three axes
    custom_lines = [Line2D([0], [0], color="#1f77b4", lw=2),
                    Line2D([0], [0], color="#ff7f0e", lw=2),
                    Line2D([0], [0], color="#2ca02c", lw=2)]
    axlist[0].legend(custom_lines, ['X', 'Y', 'Z'], bbox_to_anchor=(1.05, 1), loc='upper left')

    for ax in axlist:
        # ax.set_ylim((-np.pi,np.pi))
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=2)))
        ax.tick_params(direction="in")

    # Set equal spacing on left and right margins so axes appear in the center,
    # labels/legends appear tacked on either side, but when centered in latex, it looks better
    fig.subplots_adjust(left=0.1, right=0.9, top=1.0, hspace=0.1)

    plt.savefig("projected_rotations.pdf")
    print('aaaa')