import json
from pathlib import Path
from typing import Tuple
import os
import h5py

import git
import jax
import numpy as np
import torch
import wandb
from absl import app, flags, logging
from ml_collections import config_dict, config_flags
from ml_collections.config_dict import ConfigDict

from spatial_functa.dataloader import (
    get_augmented_dataloader,
)
from spatial_functa import SIREN, LatentVector
from spatial_functa.classifier_trainer import Trainer
from spatial_functa.grad_acc import Batch
from spatial_functa.dataloader import numpy_collate
from spatial_functa.model.mlp import MLP


_CONFIG = config_flags.DEFINE_config_file(
    "config", "experiments/downstream/config/train_functaset.py"
)
_MODEL = config_flags.DEFINE_config_file(
    "model", "experiments/downstream/config/classifier_model.py:mlp"
)
_DATASET = config_flags.DEFINE_config_file(
    "dataset", "experiments/downstream/config/functa_dataset.py:cifar10"
)
_SCHEDULER = config_flags.DEFINE_config_file(
    "scheduler", "experiments/downstream/config/scheduler.py:constant"
)


def load_config_and_store() -> Tuple[ConfigDict, Path]:
    # load the config
    config = _CONFIG.value
    model_config = _MODEL.value
    dataset_config = _DATASET.value
    scheduler_config = _SCHEDULER.value

    config.unlock()
    config.dataset = dataset_config
    config.model = model_config
    config.scheduler = scheduler_config
    config.lock()
    print(config)

    # create the experiment directory
    experiment_dir = Path(config.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # store the config
    with open(experiment_dir / "config.yaml", "w") as fp:
        json.dump(config.to_dict(), fp)

    return config, experiment_dir

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

def main(_):
    (config, experiment_dir) = load_config_and_store()

    # throw warning if the git repository has not been commited
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        logging.warning("Your git repository has uncommited changes")

    # set the random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # create the dataloaders
    path = Path(os.environ.get("DATA_PATH", "data")) / config.dataset.path

    train_dataloader = torch.utils.data.DataLoader(
        h5py_dataloader(path / "train"),
        batch_size=config.train.batch_size * jax.device_count(),
        collate_fn=batch_collate,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        h5py_dataloader(path / "val"),
        batch_size=config.train.batch_size * jax.device_count() // config.train.num_minibatches,
        collate_fn=batch_collate,
        shuffle=False,
    )

    test_dataoader = torch.utils.data.DataLoader(
        h5py_dataloader(path / "test"),
        batch_size=config.train.batch_size * jax.device_count() // config.train.num_minibatches,
        collate_fn=batch_collate,
        shuffle=False,
    )


    # create the model
    model = MLP(
        hidden_dim=config.model.hidden_dim,
        num_classes=config.dataset.num_classes,
        num_layers=config.model.num_layers,
    )


    example_batch = next(iter(train_dataloader))
    # initialize wandb
    wandb.init(
        project="classifier_spatial_functa",
        config=config.to_dict(),
        name=experiment_dir.name,
        dir=str(experiment_dir),
    )

    trainer = Trainer(
        model=model,
        example_batch=example_batch,
        config=config,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_devices=jax.device_count(),
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
