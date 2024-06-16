import torch
import numpy as np
import h5py
from spatial_functa.grad_acc import Batch
from spatial_functa.dataloader import numpy_collate

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
    def __init__(self, path):
        # npy pattern is "name_startid-endid.npy" find all files with this pattern
        paths = list(path.glob("*.h5"))
        self.num_samples = max([int(path.stem.split("_")[-1].split("-")[-1]) for path in paths])
        self.idx_to_path = {}
        self.relative_idx = {}

        for path in paths:
            start, end = path.stem.split("_")[-1].split("-")
            start, end = int(start), int(end)
            for rel_idx, idx in enumerate(range(start, end+1)):
                self.idx_to_path[idx] = path
                self.relative_idx[idx] = rel_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        path = self.idx_to_path[idx]
        rel_idx = self.relative_idx[idx]
        with h5py.File(path, "r") as f:
            return f['functabank'][rel_idx], f['labelbank'][rel_idx]

def batch_collate(batch):
    batch_list = numpy_collate(batch)
    return Batch(inputs=batch_list[0], labels=batch_list[1])