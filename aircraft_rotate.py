import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusion import ProjectedSO3Diffusion, extract
from distributions import IsotropicGaussianSO3
from models import SinusoidalPosEmb, Siren, ResLayer
from rotations import six2rmat


class ShapeNet(Dataset):
    def __init__(self, datatype, ids):
        if isinstance(ids, int):
            ids = (ids,)
        if datatype == 'train':
            filelist = 'data/shapenetcorev2_hdf5_2048/train_files.txt'
        elif datatype == 'valid':
            filelist = 'data/shapenetcorev2_hdf5_2048/val_files.txt'
        elif datatype == 'test':
            filelist = 'data/shapenetcorev2_hdf5_2048/test_files.txt'
        else:
            raise Exception(f'wrong dataset type specified: {datatype}')
        with open(filelist) as f:
            files = [x.strip('\n') for x in f.readlines()]
        self.datalist = []
        for file in files:
            with h5py.File(file, 'r') as f:
                self.datalist += [(file, i) for i, label in enumerate(f['label']) if label in ids]
        self.h5dict = dict()

    def __getitem__(self, item):
        file, idx = self.datalist[item]
        # Can't share file handles when forking to multiple processes,
        # so initialise in the __getitem__ method.
        # We also don't want to be re-opening them continuously,
        # So stick them in a dict and re-use
        try:
            f = self.h5dict[file]
        except KeyError:
            f = h5py.File(file, 'r')
            self.h5dict[file] = f
        data = f['data'][idx]

        return data

    def __len__(self):
        return len(self.datalist)


class PointCloudProj(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def forward(self, x):
        return self.data @ x.transpose(-1, -2)


class RotPredict(nn.Module):
    def __init__(self, d_model=512, nhead=8, layers=12, out_type="backprop"):
        super().__init__()
        self.out_type = out_type
        if out_type in ["skewvec", "backprop"]:
            d_out = 3
        else:
            RuntimeError(f"Unexpected out_type: {out_type}")
        self.recip_sqrt_dim = 1 / np.sqrt(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.position_siren = Siren(in_channels=3, out_channels=d_model // 2, scale=30)
        self.time_embedding = SinusoidalPosEmb(d_model // 2)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        if self.out_type == "skewvec":
            self.out_query = nn.Linear(d_model // 2, d_model)
            self.out_key = nn.Linear(d_model, d_model)
            self.out_value = nn.Linear(d_model, d_model)

        out_net = [ResLayer(nn.Sequential(nn.Linear(d_model, d_model),
                                          nn.SiLU(),
                                          )
                            )
                   for _ in range(4)]
        out_net.append(nn.Linear(d_model, d_out))
        self.out_net = nn.Sequential(*out_net)

    def forward(self, x, t):
        x_emb = self.position_siren(x)
        t_emb = self.time_embedding(t)
        t_in = torch.cat((x_emb, t_emb[:, None, :].expand(x_emb.shape)), dim=2)
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        encoding = self.encoder(t_in.transpose(0, 1)).transpose(0, 1)

        if self.out_type == "skewvec":
            # Manual single-token attention for final prediction
            Q = self.out_query(torch.ones_like(t_emb[:, None, :]))
            K = self.out_key(encoding)
            V = self.out_value(encoding)
            t_out = torch.softmax(Q @ K.transpose(-1, -2) * self.recip_sqrt_dim, dim=-1) @ V
            t_out = t_out[..., 0, :]  # Drop sequence/token dimension
        elif self.out_type == "backprop":
            t_out = encoding

        out = self.out_net(t_out)
        return out


BATCH = 2

if __name__ == "__main__":
    import wandb

    wandb.init(project='ProjectedSO3Diffusion', entity='qazwsxal')

    torch.autograd.set_detect_anomaly(True)
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    net = RotPredict().to(device)
    net.train()
    wandb.watch(net, log="all", log_freq=10)
    process = ProjectedSO3Diffusion(net).to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=1e-5)
    i = 0
    while i < 40000:
        for data in dl:
            proj = PointCloudProj(data.to(device)).to(device)
            truepos = process.identity
            loss = process(truepos.repeat(BATCH, 1, 1), proj)
            print(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            logdict = {"loss": loss.detach()}
            # Initial setup and prediction of some test data.
            if i == 0:
                with torch.no_grad():
                    t = torch.randint(0, process.num_timesteps, (BATCH,), device=device).long()
                    eps = extract(process.sqrt_one_minus_alphas_cumprod, t, t.shape)
                    noise = IsotropicGaussianSO3(eps).sample().detach()
                    x_noisy = process.q_sample(x_start=truepos.repeat(BATCH, 1, 1), t=t, noise=noise)
                    proj_x_noisy = process.projection(x_noisy)

            if i % 100 == 0:
                torch.save(net.state_dict(), "weights_aircraft.pt")

                with torch.no_grad():
                    x_recon = process.denoise_fn(proj_x_noisy, t)
                    start = proj_x_noisy[0].cpu().numpy()[::16]
                    end = (proj_x_noisy[0] - x_recon[0]).cpu().numpy()[::16]
                logdict["Predicted Gradients"] = wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": proj_x_noisy[0].cpu().numpy(),
                        "vectors": np.array([{"start": s.tolist(), "end": e.tolist()} for s, e in zip(start, end)]),
                        "boxes": np.array([{
                            "corners": [
                                [-1, -1, -1],
                                [-1,  1, -1],
                                [-1, -1,  1],
                                [ 1, -1, -1],
                                [ 1,  1, -1],
                                [-1,  1,  1],
                                [ 1, -1,  1],
                                [ 1,  1,  1]
                                ],
                            # "label": "Tree",
                            "color": [123, 321, 111],
                            }, ])
                        }
                    )
            i += 1
            wandb.log(logdict)

    torch.save(net.state_dict(), "weights_aircraft.pt")