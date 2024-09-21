# export DATA_PATH="/home/papas/project_folder/spatial_functa/functaset"
import json
from pathlib import Path
from typing import Tuple
import os
import h5py
import shutil
from distutils.dir_util import copy_tree

import git
import jax
import numpy as np
import torch
import wandb
from tqdm import tqdm
from absl import app, flags, logging
from ml_collections import config_dict, config_flags
from ml_collections.config_dict import ConfigDict

from spatial_functa.dataloader import (
    get_augmented_dataloader,
)
from spatial_functa import SIREN, LatentVector
from spatial_functa.trainer import Trainer


_CONFIG = config_flags.DEFINE_config_file(
    "config", "experiments/downstream/config/create_functaset.py"
)
_DATASET = config_flags.DEFINE_config_file(
    "dataset", "experiments/downstream/config/dataset.py:cifar10"
)
_FUNCTASET = config_flags.DEFINE_config_file(
    "functaset", "experiments/downstream/config/functaset.py"
)


def load_config_and_store() -> Tuple[ConfigDict, Path]:
    # load the config
    config = _CONFIG.value
    dataset_config = _DATASET.value
    functaset_config = _FUNCTASET.value

    config.unlock()
    config.dataset = dataset_config
    config.functaset = functaset_config
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
    functa_model_dir = config.functa_model_dir
    with open(Path(functa_model_dir) / "config.yaml", "r") as fp:
        loaded_config = ConfigDict(json.load(fp))

    # throw warning if the git repository has not been commited
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        logging.warning("Your git repository has uncommited changes")

    # set the random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # torch.backends.cudnn.deterministic = True

    # create the dataloaders
    train_dataloader = get_augmented_dataloader(
        dataset_config=config.dataset,
        subset="train",
        shuffle=False,
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
        input_dim=loaded_config.dataset.resolution**2,
        image_width=loaded_config.dataset.resolution,
        image_height=loaded_config.dataset.resolution,
        num_layers=loaded_config.model.num_layers,
        hidden_dim=loaded_config.model.hidden_dim,
        output_dim=loaded_config.dataset.num_channels,
        latent_dim=loaded_config.model.latent_dim,
        latent_spatial_dim=loaded_config.model.latent_spatial_dim,
        omega_0=loaded_config.model.omega_0,
        modulation_hidden_dim=loaded_config.model.modulation_hidden_dim,
        modulation_num_layers=loaded_config.model.modulation_num_layers,
        learn_lrs=loaded_config.model.learn_lrs,
        lr_init_range=loaded_config.train.inner_lr_init_range,
        lr_clip_range=loaded_config.train.inner_lr_clip_range,
        scale_modulate=loaded_config.model.scale_modulate,
        shift_modulate=loaded_config.model.shift_modulate,
        interpolation_type=loaded_config.model.interpolation_type,
    )

    latent_vector_model = LatentVector(
        latent_dim=loaded_config.model.latent_dim,
        latent_spatial_dim=loaded_config.model.latent_spatial_dim,
    )

    example_batch = next(iter(train_dataloader))
    # initialize wandb

    trainer = Trainer(
        model=model,
        latent_vector_model=latent_vector_model,
        example_batch=example_batch,
        config=loaded_config,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_devices=jax.device_count(),
    )

    trainer.load(
        str(
            Path(
                (loaded_config.train.checkpointing.checkpoint_dir) / Path("best_psnr")
            ).absolute()
        )
    )

    field_params = trainer.state.params
    starting_latent_params = trainer.latent_params
    functaset_dir = Path(os.environ.get("DATA_PATH", "data")) / Path(
        config.functaset.path
    )
    functaset_dir.mkdir(parents=True, exist_ok=True)

    # copy the loaded_config to the functaset_dir
    with open(functaset_dir / "config.yaml", "w") as fp:
        json.dump(loaded_config.to_dict(), fp)

    # copy the model checkpoint
    original_checkpoint_dir = Path(
        (loaded_config.train.checkpointing.checkpoint_dir) / Path("best_psnr")
    ).absolute()
    destination_checkpoint_dir = functaset_dir / "model_checkpoint"
    destination_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(str(original_checkpoint_dir), str(destination_checkpoint_dir))

    def make_functaset(loader, split):
        wandb.init(
            project="spatial_functa_create_functaset",
            config=config.to_dict(),
            name=experiment_dir.name + f"_{split}",
            dir=str(experiment_dir),
        )
        functaset_size = len(loader.dataset)
        functaset_file = h5py.File(
            functaset_dir / f"functaset_{config.functaset.name}_{split}.h5", "w"
        )
        functaset_dset = functaset_file.create_dataset(
            "functaset",
            shape=(
                functaset_size,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_dim,
            ),
            chunks=(
                1,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_dim,
            ),
        )
        labels_dset = functaset_file.create_dataset(
            "labels",
            shape=(functaset_size, 1),
        )
        functaset_idx = 0
        tqdm_iter = tqdm(
            enumerate(loader), total=len(loader), desc=f"Creating functaset {split}"
        )

        # loop through the training dataset
        for i, batch in tqdm_iter:
            batch = trainer.process_batch(batch)
            coords = batch.inputs
            target = batch.targets
            labels = batch.labels
            loss, recon, latent_params = trainer.forward(
                field_params, starting_latent_params, coords, target
            )

            latent_vector = trainer.get_latent_vector(latent_params)
            latent_vector = latent_vector.reshape(
                -1,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_spatial_dim,
                loaded_config.model.latent_dim,
            )
            labels = labels.reshape(-1, 1)

            num_vectors = latent_vector.shape[0]
            assert np.any(
                latent_vector
                != trainer.get_latent_vector(starting_latent_params).reshape(
                    -1,
                    loaded_config.model.latent_spatial_dim,
                    loaded_config.model.latent_spatial_dim,
                    loaded_config.model.latent_dim,
                )
            ), "Nothing changed in the latent vector"

            functaset_dset[functaset_idx : functaset_idx + num_vectors] = (
                jax.device_get(latent_vector)
            )
            labels_dset[functaset_idx : functaset_idx + num_vectors] = jax.device_get(
                labels
            )

            functaset_idx += num_vectors

            if i % config.train.log_steps.loss == 0:
                wandb.log(
                    {"loss": loss, "functaset_idx": functaset_idx, "i": i}, step=i
                )
                # print to tqdm
                tqdm_iter.set_postfix(
                    {
                        "loss": loss.mean(),
                        "functaset_idx": functaset_idx,
                    }
                )

            if i % config.train.log_steps.image == 0:
                recon = jax.device_get(recon).reshape(-1, *trainer.image_shape)
                target = jax.device_get(target).reshape(-1, *trainer.image_shape)

                for j in range(min(5, trainer.num_signals_per_device)):
                    wandb.log(
                        {
                            f"images/recon_{j}": wandb.Image(recon[j]),
                            f"images/target_{j}": wandb.Image(target[j]),
                        },
                        step=i,
                    )
        functaset_file.close()
        wandb.finish()

    make_functaset(train_dataloader, "train")
    make_functaset(val_dataloader, "val")
    make_functaset(test_dataloader, "test")


if __name__ == "__main__":
    app.run(main)
