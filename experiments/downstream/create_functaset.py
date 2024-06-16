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


def load_config_and_store() -> Tuple[ConfigDict, Path]:
    # load the config
    config = _CONFIG.value
    dataset_config = _DATASET.value

    config.unlock()
    config.dataset = dataset_config
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

    trainer.load(str(Path((loaded_config.train.checkpointing.checkpoint_dir)/Path("best_psnr")).absolute()))

    field_params = trainer.state.params
    starting_latent_params = trainer.latent_params

    batch_size = config.train.batch_size * jax.device_count() * config.train.num_minibatches
    
    def make_functaset(loader, name, batch_size):

        wandb.init(
            project="spatial_functa_create_dataset",
            config=config.to_dict(),
            name=experiment_dir.name + f"_{name}",
            dir=str(experiment_dir),
        )
        functabank_size = min(config.functaset.functa_bank_size_per_batch * batch_size, len(loader.dataset))
        functabank = np.zeros((functabank_size, loaded_config.model.latent_spatial_dim, loaded_config.model.latent_spatial_dim, loaded_config.model.latent_dim))
        labelbank = np.zeros((functabank_size, 1))

        functabank_idx = 0
        real_idx = 0
        prev_idx = 0

        functaset_dir = Path(os.environ.get("DATA_PATH", "data")) / Path(config.functaset.path) / Path(name)
        functaset_dir.mkdir(parents=True, exist_ok=True)

        # loop through the training dataset
        for i, batch in tqdm(enumerate(loader)):
            batch = trainer.process_batch(batch)
            coords = batch.inputs
            target = batch.targets
            labels = batch.labels
            loss, recon, latent_params = trainer.forward(field_params, starting_latent_params, coords, target)

            latent_vector = trainer.get_latent_vector(latent_params)
            latent_vector = latent_vector.reshape(-1, loaded_config.model.latent_spatial_dim, loaded_config.model.latent_spatial_dim, loaded_config.model.latent_dim)
            labels = labels.reshape(-1, 1)

            num_vectors = latent_vector.shape[0]

            functabank[functabank_idx:functabank_idx+num_vectors] = jax.device_get(latent_vector)
            labelbank[functabank_idx:functabank_idx+num_vectors] = jax.device_get(labels)

            functabank_idx += num_vectors
            real_idx += num_vectors

            if i % config.train.log_steps.loss == 0:
                wandb.log({
                    "loss": loss,
                    "functabank_idx": functabank_idx,
                    "real_idx": real_idx,  
                    "i": i    
                    }, step=i)

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

            if functabank_idx == functabank_size:
                with h5py.File(functaset_dir / f"functabank_{prev_idx}-{real_idx}.h5", 'w') as f:
                    f.create_dataset('functabank', data=functabank)
                    f.create_dataset('labelbank', data=labelbank)

                prev_idx = real_idx
                functabank_idx = 0
        
        if functabank_idx > 0:
            with h5py.File(functaset_dir / f"functabank_{prev_idx}-{real_idx}.h5", 'w') as f:
                f.create_dataset('functabank', data=functabank)
                f.create_dataset('labelbank', data=labelbank)
        
        wandb.finish()
        

    make_functaset(train_dataloader, "train", batch_size)
    make_functaset(val_dataloader, "val", batch_size)
    make_functaset(test_dataloader, "test", batch_size)



if __name__ == "__main__":
    app.run(main)
