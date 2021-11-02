from mayavi import mlab
import matplotlib.pyplot as plt
from colors import *
from util import *
import torch
import torch.nn as nn
from math import pi

R_1_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(1 * pi / 3), torch.tensor(0.0))
R_2_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(2 * pi / 3), torch.tensor(0.0))


points = 1000
weights = torch.linspace(0, 1, points)

distrib = so3_lerp(R_1_1[None], R_2_1[None], weights[:, None])

x, y, z = rmat_to_euler(distrib)
distrib_back = euler_to_rmat(x,y,z)

fig, axlist = plt.subplots(nrows=3, ncols=1, sharex=True)
axlist[0].plot(x, c=BLUE)
axlist[1].plot(y, c=ORANGE)
axlist[2].plot(z, c=GREEN)
plt.show()

# Define sphere
count = 101
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
x_p, y_p, z_p = torch.unbind(distrib, dim=2)
opts = dict(resolution=20, transparent=True, opacity=1.0, scale_mode='none', scale_factor=0.07)
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
