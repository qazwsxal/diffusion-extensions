import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusion import ProjectedSO3Diffusion
from models import SinusoidalPosEmb, Siren
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
    def __init__(self, d_model=512, nhead=4, layers=6):
        super().__init__()
        self.recip_sqrt_dim = 1 / np.sqrt(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.position_siren = Siren(in_channels=3, out_channels=d_model, scale=30, post_scale=True)
        self.time_embedding = SinusoidalPosEmb(d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.out_query = nn.Linear(d_model, d_model)
        self.out_key = nn.Linear(d_model, d_model)
        self.out_value = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, 6)

    def forward(self, x, t):
        x_emb = self.position_siren(x)
        # Transpose batch and sequence dimension
        # Because we're using a version of PT that doesn't support
        # Batch first.
        x_enc = self.encoder(x_emb.transpose(0, 1)).transpose(0, 1)
        t_emb = self.time_embedding(t)

        # Manual single-token attention for final prediction
        Q = self.out_query(t_emb)
        K = self.out_key(x_enc)
        V = self.out_value(x_enc)
        t_out = torch.softmax(Q @ K.transpose(-1, -2) * self.recip_sqrt_dim, dim=-1) @ V
        out = self.out_linear(F.elu(t_out))
        out = out[..., 0, :]  # Drop sequence/token dimension
        return six2rmat(out)


BATCH = 4

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    ds = ShapeNet('train', (0,))
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    net = RotPredict().to(device)
    process = ProjectedSO3Diffusion(net).to(device)
    optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)
    i = 0
    while i<40000:
        for data in dl:
            i += 1
            proj = PointCloudProj(data.to(device)).to(device)
            truepos = process.identity
            loss = process(truepos.repeat(BATCH, 1, 1), proj)
            print(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i == 40000:
                break
    torch.save(net.state_dict(), "weights_aircraft.pt")