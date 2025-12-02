import json
from pathlib import Path
from typing import Tuple

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
from spatial_functa.trainer import Trainer


_CONFIG = config_flags.DEFINE_config_file(
    "config", "experiments/fitting/config/config.py"
)
_MODEL = config_flags.DEFINE_config_file(
    "model", "experiments/fitting/config/model.py:latent"
)
_DATASET = config_flags.DEFINE_config_file(
    "dataset", "experiments/fitting/config/dataset.py:cifar10"
)
_SCHEDULER = config_flags.DEFINE_config_file(
    "scheduler", "experiments/fitting/config/scheduler.py:constant"
)


def load_config() -> ConfigDict:
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

    return config


def main(_):
    config = load_config()

    # throw warning if the git repository has not been commited
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        logging.warning("Your git repository has uncommited changes")
    # initialize wandb
    wandb.init(
        project="spatial_functa",
        config=config.to_dict(),
        name=config.experiment_name,
        dir=config.experiments_root,
    )
    run_id = wandb.run.id
    experiment_dir = Path(config.experiments_root) / Path(config.experiment_name) / Path(run_id)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # store the config
    with open(experiment_dir / "config.yaml", "w") as fp:
        json.dump(config.to_dict(), fp)

    # set the random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # torch.backends.cudnn.deterministic = True

    # create the dataloaders
    train_dataloader = get_augmented_dataloader(
        dataset_config=config.dataset,
        subset="train",
        shuffle=True,
        seed=config.seed,
        batch_size=config.train.batch_size,
        num_minibatches=config.train.num_minibatches,
    )

    val_dataloader = get_augmented_dataloader(
        dataset_config=config.dataset,
        subset="val",
        shuffle=False,
        seed=config.seed,
        batch_size=config.train.batch_size,
        num_minibatches=config.train.num_minibatches,
    )

    test_dataloader = get_augmented_dataloader(
        dataset_config=config.dataset,
        subset="test",
        shuffle=False,
        seed=config.seed,
        batch_size=config.train.batch_size,
        num_minibatches=config.train.num_minibatches,
    )

    # create the model
    model = SIREN(
        input_dim=config.dataset.resolution**2,
        image_width=config.dataset.resolution,
        image_height=config.dataset.resolution,
        num_layers=config.model.num_layers,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.dataset.num_channels,
        latent_dim=config.model.latent_dim,
        latent_spatial_dim=config.model.latent_spatial_dim,
        omega_0=config.model.omega_0,
        modulation_hidden_dim=config.model.modulation_hidden_dim,
        modulation_num_layers=config.model.modulation_num_layers,
        learn_lrs=config.model.learn_lrs,
        lr_init_range=config.train.inner_lr_init_range,
        lr_clip_range=config.train.inner_lr_clip_range,
        scale_modulate=config.model.scale_modulate,
        shift_modulate=config.model.shift_modulate,
        interpolation_type=config.model.interpolation_type,
    )

    latent_vector_model = LatentVector(
        latent_dim=config.model.latent_dim,
        latent_spatial_dim=config.model.latent_spatial_dim,
    )

    example_batch = next(iter(train_dataloader))

    trainer = Trainer(
        model=model,
        latent_vector_model=latent_vector_model,
        example_batch=example_batch,
        config=config,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        experiment_dir=experiment_dir,
        num_devices=jax.device_count(),
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    app.run(main)
