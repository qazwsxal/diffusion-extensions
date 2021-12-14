import h5py
from torch.utils.data import Dataset
import torch

class ShapeNet(Dataset):
    def __init__(self, datatype, ids, samples=None):
        self.samples = samples
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
        self.probs = torch.ones((2048))/2048

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
        data = torch.tensor(f['data'][idx])
        if self.samples is not None:
            data = data[torch.multinomial(self.probs, num_samples=self.samples)]
        return data

    def __len__(self):
        return len(self.datalist)