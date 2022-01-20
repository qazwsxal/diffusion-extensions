import torch
import matplotlib.pyplot as plt

eul = torch.load("weights/results_aircraft_eul.pt", map_location=torch.device("cpu"))
eul_flat = eul.ravel()
eul_flat_sort, _ = torch.sort(eul_flat)

so3 = torch.load("weights/results_aircraft_so3.pt", map_location=torch.device("cpu"))
so3_flat = so3.ravel()
so3_flat_sort, _ = torch.sort(so3_flat)

plt.plot(eul_flat_sort.numpy())
plt.plot(so3_flat_sort.numpy())
plt.show()
eul_mean = eul.mean(dim=1)
eul_std = eul.std(dim=1)

plt.scatter(eul_mean, eul_std)
plt.show()
print('aaa')