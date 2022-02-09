import torch
import matplotlib.pyplot as plt

eul = torch.load("weights/results_aircraft_eul.pt", map_location=torch.device("cpu"))
eul_flat = eul.ravel()
eul_flat_sort, _ = torch.sort(eul_flat)

so3 = torch.load("weights/results_aircraft_so3.pt", map_location=torch.device("cpu"))
so3_flat = so3.ravel()
so3_flat_sort, _ = torch.sort(so3_flat)

plt.plot(eul_flat_sort.numpy(), label="euler")
plt.plot(so3_flat_sort.numpy(), label="so3")
plt.legend()
plt.show()

count = len(so3_flat_sort)
print(count)
percentiles = (0.01, 0.05, 0.10, 0.50, 0.90, 0.95,  0.99)
per_idxs = [int(count * p) for p in percentiles]
print("percentiles", *[f" & {p:.0%}" for p in  percentiles],r"\\")
print("euler", *[f" & {eul_flat_sort[i].item():.2f}" for i in per_idxs],r"\\")
print("so3", *[f" & {so3_flat_sort[i].item():.2f}" for i in per_idxs],r"\\")
