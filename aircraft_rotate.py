from diffusion import ProjectedSO3Diffusion
import torch
import numpy as np
import torch
from models import SinusoidalPosEmb
from diffusion import ProjectedGaussianDiffusion
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import torch.nn as nn
import torch.nn.functional as F

from util import to_device



import h5py
import numpy as np

class ShapeNet(Dataset):
    def __init__(self, type, ids):
        if type == 'train':
            filelist = 'data/shapenetcorev2_hdf5_2048/train_files.txt'
        elif type == 'valid':
            filelist = 'data/shapenetcorev2_hdf5_2048/val_files.txt'
        elif type == 'test':
            filelist = 'data/shapenetcorev2_hdf5_2048/test_files.txt'
        with open(filelist) as f:
            files = f.readlines()
        print('aaa')


a = ShapeNet('train', 0)