import numpy as np
import torch
from euler_lock_train import EulerRotPredict
from mpl_utils import *
from tqdm import tqdm
from rotations import *

from diffusion import GaussianDiffusion

BATCH = 512

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = EulerRotPredict().to(device)
    net.load_state_dict(torch.load("weights_euler_lock.pt", map_location=device))
    net.eval()
    process = GaussianDiffusion(net, loss_type="skewvec").to(device)
    with torch.no_grad():
        # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
        R, _ = torch.qr(torch.randn((BATCH, 3, 3)))
        res = torch.zeros((process.num_timesteps, BATCH, 3, 3))
        for i in tqdm(reversed(range(0, process.num_timesteps)),
                      desc='sampling loop time step',
                      total=process.num_timesteps,
                      ):
            res[i] = R.detach()
            R = process.p_sample(R, torch.full((1,), i, device=device, dtype=torch.long))

    # Decompose into euler-angle form for plotting.
    x, y, z = rmat_to_euler(res)

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
    axlist[0].legend(custom_lines, ['X', 'Y', 'Z'])

    for ax in axlist:
        # ax.set_ylim((-np.pi,np.pi))
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=2)))
        ax.tick_params(direction="in")

    # Set equal spacing on left and right margins so axes appear in the center,
    # labels/legends appear tacked on either side, but when centered in latex, it looks better
    fig.subplots_adjust(left=0.1, right=0.9, top=1.0, hspace=0.1)

    plt.show()
    plt.savefig("rotations.eps")
    out = res[0]
    z90 = torch.tensor([[0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])
    z90pangle = rmat_dist(res, z90[None, None]) * 0.70710678118
    z90mangle = rmat_dist(res, z90.T[None, None]) * 0.70710678118
    z90close = (z90pangle[0] < z90mangle[0])
    z90stack = torch.stack((z90mangle, z90pangle), dim=0)
    z90best = torch.gather(z90stack, 0, z90close.expand(1, *z90stack.shape[1:]).to(int))[0]
    plt.plot(torch.arange(1000).flip(0), z90best, alpha=0.2, c="#1f77b4")
    print('aaaaa')