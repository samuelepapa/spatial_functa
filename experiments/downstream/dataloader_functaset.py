import torch
import numpy as np
import h5py
from spatial_functa.grad_acc import Batch
from spatial_functa.dataloader import numpy_collate

import pdb

from pathlib import Path

class npy_dataloader(torch.utils.data.Dataset):
    def __init__(self, path):
        # npy pattern is "name_startid-endid.npy" find all files with this pattern
        paths = list(path.glob("*.npy"))
        functabanks = []
        for path in paths:
            functabanks.append(np.load(path))

        self.functaset = np.concatenate(functabanks, axis=0)
        

    def __len__(self):
        return 16 # len(self.functaset)

    def __getitem__(self, idx):
        return self.functaset[idx]
    
class h5py_dataloader(torch.utils.data.Dataset):
    def __init__(self, path, name, split):
        # npy pattern is "name_startid-endid.npy" find all files with this pattern
        self.path = path / Path(f"functaset_{name}_{split}.h5")
        with h5py.File(self.path, "r") as f:
            functaset = f['functaset']
            
            self.num_samples = functaset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            return f['functaset'][idx], f['labels'][idx]


def batch_collate(batch):
    batch_list = numpy_collate(batch)
    return Batch(inputs=batch_list[0], labels=batch_list[1])