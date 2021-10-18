from mayavi import mlab
import numpy as np
import torch
from so3_lock_train import RotPredict
from mpl_utils import *
from tqdm import tqdm
from rotations import *
from colors import *

from diffusion import SO3Diffusion

BATCH = 64

if __name__ == "__main__":


    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = RotPredict(out_type="skewvec").to(device)
    net.load_state_dict(torch.load("weights/weights_so3_lock.pt", map_location=device))
    net.eval()
    process = SO3Diffusion(net, loss_type="skewvec").to(device)
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


    endmats = res[0]

    # Define sphere
    count = 101
    pi = np.pi
    cos = torch.cos
    sin = torch.sin
    phi = torch.linspace(0, pi, count)
    theta = torch.linspace(0, 2 * pi, count)

    phi, theta = torch.meshgrid(phi, theta)
    x = sin(phi) * cos(theta)
    y = sin(phi) * sin(theta)
    z = cos(phi)
    points = torch.stack((x, y, z), dim=0)
    axes = torch.eye(3)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))
    mlab.clf()
    obj = mlab.mesh(x.numpy(),
                    y.numpy(),
                    z.numpy(),
                    color=(0.9, 0.9, 0.9),
                    opacity=0.2,
                    transparent=True,
                    )
    # Draw axes

    mlab.plot3d([0, 1], [0, 0], [0, 0], color=GREY_F, tube_radius=None)
    mlab.plot3d([0, 0], [0, 1], [0, 0], color=GREY_F, tube_radius=None)
    mlab.plot3d([0, 0], [0, 0], [0, 1], color=GREY_F, tube_radius=None)
    mlab.text3d(0.7, 0, 0, 'X', color=BLUE_F, scale=0.05)
    mlab.text3d(0, 0.7, 0, 'Y', color=ORANGE_F, scale=0.05)
    mlab.text3d(0, 0, 0.7, 'Z', color=GREEN_F, scale=0.05)
    x_p, y_p, z_p = torch.unbind(endmats, dim=2)
    opts = dict(resolution=20,transparent=True, opacity=1.0, scale_mode='none', scale_factor=0.07)
    mlab.points3d(*x_p.T, color=BLUE_F, **opts)
    mlab.points3d(*y_p.T, color=ORANGE_F, **opts)
    mlab.points3d(*z_p.T, color=GREEN_F, **opts)
    mlab.view(azimuth=60,
              elevation=60,
              distance=6.2,
              focalpoint=(0.0, 0.0, 0.0),
              )

    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 1.0
    mlab.show()
    print('aaaa')
