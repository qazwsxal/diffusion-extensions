import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import *
import torch
import torch.nn as nn
from math import pi

R_1_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(1 * pi / 3), torch.tensor(0.0))
R_2_1 = euler_to_rmat(torch.tensor(0.0), torch.tensor(2 * pi / 3), torch.tensor(0.0))


points = 1000
weights = torch.linspace(0, 1, points)

distrib = so3_lerp(R_1_1[None], R_2_1[None], weights[:, None])

x, y, z = rmat_to_euler(distrib)

fig, axlist = plt.subplots(nrows=3, ncols=1, sharex=True)
axlist[0].plot(x, c="#1f77b4")
axlist[1].plot(y, c="#ff7f0e")
axlist[2].plot(z, c="#2ca02c")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*distrib[:, :, 0].T, c="#1f77b4")
ax.scatter(*distrib[:, :, 1].T, c="#ff7f0e")
ax.scatter(*distrib[:, :, 2].T, c="#2ca02c")
plt.show()

width = 50
smooth = nn.Conv1d(1, 1, width)
smooth.weight.data = torch.ones_like(smooth.weight.data) / width
smooth.bias.data = torch.zeros_like(smooth.bias.data)
with torch.no_grad():
    x_smooth = smooth(x[None, None] + 0.05 * torch.randn_like(x[None, None]))[0, 0]
    y_smooth = smooth(y[None, None] + 0.05 * torch.randn_like(y[None, None]))[0, 0]
    z_smooth = smooth(z[None, None] + 0.05 * torch.randn_like(z[None, None]))[0, 0]

fig, axlist = plt.subplots(nrows=3, ncols=1, sharex=True)
axlist[0].plot(x_smooth, c="#1f77b4")
axlist[1].plot(y_smooth, c="#ff7f0e")
axlist[2].plot(z_smooth, c="#2ca02c")
plt.show()

distrib_euler_smooth = euler_to_rmat(x_smooth, y_smooth, z_smooth)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*distrib_euler_smooth[:, :, 0].T, c="#1f77b4")
ax.plot(*distrib_euler_smooth[:, :, 1].T, c="#ff7f0e")
ax.plot(*distrib_euler_smooth[:, :, 2].T, c="#2ca02c")
plt.show()
print('aaa')