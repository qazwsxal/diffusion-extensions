from mayavi import mlab
import numpy as np
from PIL import Image
import distributions
import torch

# Set up colour maps:
VMAX = 15.0
VMIN = -7.0
colours = np.array(((255, 127,  14, 0),
                    ( 44, 160,  44, 0),
                    ( 31, 119, 180, 0),
                    ))[:,None].repeat(256,axis=1)
colours[:,:, -1] = np.linspace(0, 255, 256)



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
epsilons = torch.logspace(-2, 0.5, 6)
for eps in epsilons:
    print(eps.item())
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
    dist = distributions.IsotropicGaussianSO3(eps)
    mlab.clf()
    for i in range(3):
        axis = axes[i, :, None, None]
        angles = (points * axis).sum(dim=0).acos()
        probs = dist._eps_ft(angles).abs().log()
        probs[probs.isinf()] = VMIN
        print(probs.max().item(), probs.min().item())

        obj = mlab.mesh(x.numpy(),
                        y.numpy(),
                        z.numpy(),
                        scalars=probs.numpy(),
                        colormap='jet',
                        vmax = VMAX,
                        vmin = VMIN,
                        )
        obj.module_manager.scalar_lut_manager.lut.table = colours[i]
    mlab.view(azimuth=60,
              elevation=60,
              distance=6.2,
              focalpoint=(0.0, 0.0, 0.0),
              )
    mlab.orientation_axes()
    mlab.show()
mlab.close(all=True)