import pickle

import matplotlib.pyplot as plt
import torch

import util

se3_samples = pickle.load(open("prot_samples_se3.pkl", "rb"))
eul_samples = pickle.load(open("prot_samples_eul.pkl", "rb"))

se3_rots = torch.cat([s.rot for samp in se3_samples for s in samp], dim=0)
se3_shift = torch.cat([s.shift for samp in se3_samples for s in samp], dim=0)

eul_rots = torch.cat([s.rot for samp in eul_samples for s in samp], dim=0)
eul_shift = torch.cat([s.shift for samp in eul_samples for s in samp], dim=0)

trueshift = torch.zeros((1, 3))
truerot = torch.eye(3)[None]

se3_dists = se3_shift.norm(dim=1)
eul_dists = eul_shift.norm(dim=1)

_, se3_angles = util.rmat_to_aa(se3_rots)
_, eul_angles = util.rmat_to_aa(eul_rots)
print('aaaa')

se3_angle_sort, _ = torch.sort(se3_angles.squeeze())
eul_angle_sort, _ = torch.sort(eul_angles.squeeze())
plt.plot(eul_angle_sort.numpy(), label="euler")
plt.plot(se3_angle_sort.numpy(), label="se3")
plt.legend()
plt.show()

se3_dist_sort, _ = torch.sort(se3_dists.squeeze())
eul_dist_sort, _ = torch.sort(eul_dists.squeeze())
plt.plot(eul_dist_sort.numpy(), label="euler")
plt.plot(se3_dist_sort.numpy(), label="se3")
plt.legend()
plt.show()

count = len(se3_angle_sort)
print(count)
percentiles = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99)
per_idxs = [int(count * p) for p in percentiles]
print("percentiles", *[f" & {p:.0%}" for p in  percentiles],r"\\")
print("euler", *[f" & {eul_angle_sort[i].item():.2f}" for i in per_idxs],r"\\")
print("so3", *[f" & {se3_angle_sort[i].item():.2f}" for i in per_idxs],r"\\")
print('------')
print("percentiles", *[f" & {p:.0%}" for p in  percentiles],r"\\")
print("euler", *[f" & {eul_dist_sort[i].item():.2f}" for i in per_idxs],r"\\")
print("so3", *[f" & {se3_dist_sort[i].item():.2f}" for i in per_idxs],r"\\")
