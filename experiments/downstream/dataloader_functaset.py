from typing import Literal
import torch
import numpy as np
import h5py
from spatial_functa.grad_acc import Batch
from spatial_functa.dataloader import numpy_collate

import pdb
from absl import logging

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
    def __init__(self, path:Path, name:str, split:Literal['train', 'val', 'test'], preload:bool=True):
        # npy pattern is "name_startid-endid.npy" find all files with this pattern
        self.path = path / Path(f"functaset_{name}_{split}.h5")
        self.preload = preload
        if self.preload:
            with h5py.File(self.path, "r") as f:
                self.functaset = f['functaset'][...]
                self.labels = f['labels'][...]
                self.num_samples = self.functaset.shape[0]
        else:
            with h5py.File(self.path, "r") as f:
                self.num_samples = f['functaset'].shape[0]
        logging.info(f"Loaded {self.num_samples} samples from {self.path}")            

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preload:
            return self.functaset[idx], self.labels[idx]
        else:
            with h5py.File(self.path, "r") as f:
                return f['functaset'][idx], f['labels'][idx]


def batch_collate(batch):
    batch_list = numpy_collate(batch)
    return Batch(inputs=batch_list[0], labels=batch_list[1])