import numpy as np
from distributions import Bingham
import torch
import os
# assert(os.environ['ETS_TOOLKIT'] == 'qt4')
from mayavi import mlab
from bingham_train import covpairs, loc
from util import quat_to_rmat

BATCH=1024

from colors import *

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))

for name, acro, cov in covpairs:
    print(name)
    quats = Bingham(loc, covariance_matrix=cov).sample((BATCH,))
    rotmats = quat_to_rmat(quats)
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
    x_p, y_p, z_p = torch.unbind(rotmats, dim=2)
    opts = dict(resolution=20, transparent=False, opacity=1.0, scale_mode='none', scale_factor=0.07)
    mlab.points3d(*x_p.T, color=BLUE_F, **opts)
    mlab.points3d(*y_p.T, color=ORANGE_F, **opts)
    mlab.points3d(*z_p.T, color=GREEN_F, **opts)
    mlab.view(azimuth=60,
              elevation=60,
              distance=6.2,
              focalpoint=(0.0, 0.0, 0.0),
              )

    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 1.05
    mlab.savefig(f'images/{acro}.png')
    # mlab.show()
    mlab.clf()
name, acro, cov = covpairs[0]
print(name)
quats = Bingham(loc, covariance_matrix=cov).sample((BATCH,))
rotmats = quat_to_rmat(quats)
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
x_p, y_p, z_p = torch.unbind(rotmats, dim=2)
opts = dict(resolution=20, transparent=False, opacity=1.0, scale_mode='none', scale_factor=0.07)
mlab.points3d(*x_p.T, color=BLUE_F, **opts)
mlab.points3d(*y_p.T, color=ORANGE_F, **opts)
mlab.points3d(*z_p.T, color=GREEN_F, **opts)
mlab.view(azimuth=60,
          elevation=60,
          distance=6.2,
          focalpoint=(0.0, 0.0, 0.0),
          )

mlab.gcf().scene.parallel_projection = True
mlab.gcf().scene.camera.parallel_scale = 1.05
mlab.savefig(f'images/{acro}.png')
# mlab.show()
mlab.clf()
mlab.close(all=True)