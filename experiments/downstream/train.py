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
from spatial_functa.model.transformer import VisionTransformer
from dataloader_functaset import h5py_dataloader, batch_collate


_CONFIG = config_flags.DEFINE_config_file(
    "config", "experiments/downstream/config/train_functaset.py"
)
_MODEL = config_flags.DEFINE_config_file(
    "model", "experiments/downstream/config/classifier_model.py:transformer"
)
_FUNCTASET = config_flags.DEFINE_config_file(
    "functaset", "experiments/downstream/config/functaset.py"
)
_SCHEDULER = config_flags.DEFINE_config_file(
    "scheduler", "experiments/downstream/config/scheduler.py:constant"
)


def load_config_and_store() -> Tuple[ConfigDict, Path]:
    # load the config
    config = _CONFIG.value
    model_config = _MODEL.value
    functaset_config = _FUNCTASET.value
    scheduler_config = _SCHEDULER.value

    config.unlock()
    config.functaset = functaset_config
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
    path = Path(os.environ.get("DATA_PATH", "data")) / config.functaset.path
    name = config.functaset.name
    train_dataloader = torch.utils.data.DataLoader(
        h5py_dataloader(path, name=name, split="train"),
        batch_size=config.train.batch_size * jax.device_count(),
        collate_fn=batch_collate,
        shuffle=True,
        drop_last=True,
        num_workers=config.functaset.num_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        h5py_dataloader(path, name=name, split="val"),
        batch_size=config.train.batch_size
        * jax.device_count()
        // config.train.num_minibatches,
        collate_fn=batch_collate,
        shuffle=False,
        drop_last=True,
        num_workers=config.functaset.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        h5py_dataloader(path, name=name, split="test"),
        batch_size=config.train.batch_size
        * jax.device_count()
        // config.train.num_minibatches,
        collate_fn=batch_collate,
        shuffle=False,
        drop_last=True,
        num_workers=config.functaset.num_workers,
    )

    # create the model
    if config.model.name == "mlp":
        model = MLP(
            hidden_dim=config.model.hidden_dim,
            num_classes=config.functaset.num_classes,
            num_layers=config.model.num_layers,
            dropout_prob=config.model.dropout_prob,
        )
    elif config.model.name == "transformer":
        model = VisionTransformer(
            num_classes=config.functaset.num_classes,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            num_patches=config.model.num_patches,
            dropout_prob=config.model.dropout_prob,
        )
    else:
        raise ValueError(f"Unknown model {config.model.name}")

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
