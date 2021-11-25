import torch
import matplotlib.pyplot as plt

a = torch.load("weights/angles.pt", map_location=torch.device("cpu"))
a_flat = a.ravel()
a_flat_sort, _ = torch.sort(a_flat)

plt.plot(a_flat_sort.numpy())
plt.show()
a_mean = a.mean(dim=1)
a_std = a.std(dim=1)

plt.scatter(a_mean, a_std)
plt.show()
print('aaa')