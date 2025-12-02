import dataclasses
import math
import pdb
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import flax
import jax
import numpy as np
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax._src.typing import Array
from jax.nn.initializers import Initializer
from tqdm import tqdm

KeyArray = Array
from functools import partial
from itertools import cycle
from pathlib import Path

import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb
from absl import logging
from flax import linen as nn
from flax.training import train_state
from jax import core, dtypes, random
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

from spatial_functa.grad_acc import (
    Batch,
    TrainState,
    accumulate_gradients,
    default_get_minibatch,
)
from spatial_functa.metrics import mse, psnr
from spatial_functa.model.spatial_functa import SIREN
from spatial_functa.nn_utils import extract_learning_rate, set_profiler
from spatial_functa.opt_builder import build_lr_scheduler

Array = jnp.ndarray


def forward(latent_params, field_params, model, latent_model, coords):
    latent_vector = latent_model.apply(
        {"params": latent_params},
    )
    pred = model.apply({"params": field_params}, coords, latent_vector)
    return pred


mse_fn = jax.jit(lambda x, y: jnp.mean(jnp.square(x - y)))


def loss_fn_image(latent_params, field_params, model, latent_model, coords, target):
    pred = forward(latent_params, field_params, model, latent_model, coords)
    return mse_fn(pred, target), pred


@jax.named_scope("inner_fit")
def inner_fit(
    field_params,
    latent_params,
    model,
    latent_model,
    opt_inner: optax.GradientTransformation,
    coords,
    inner_steps,
    target,
) -> Tuple[Array, Array, Array]:
    if "lrs" in field_params.keys():
        opt_inner = optax.scale_by_learning_rate(field_params["lrs"]["lrs"])
        opt_inner_state = opt_inner.init(latent_params)
    else:
        opt_inner_state = opt_inner.init(latent_params)

    for _ in range(inner_steps):
        # run the inner optimization
        (loss, recon), grads = jax.value_and_grad(loss_fn_image, has_aux=True)(
            latent_params, field_params, model, latent_model, coords, target
        )
        latent_updates, opt_inner_state = opt_inner.update(grads, opt_inner_state)

        latent_params = optax.apply_updates(latent_params, latent_updates)

    # final loss computation
    loss, recon = loss_fn_image(
        latent_params, field_params, model, latent_model, coords, target
    )

    return loss, recon, latent_params


class Trainer:
    def __init__(
        self,
        model,
        latent_vector_model,
        example_batch,
        config,
        train_loader,
        val_loader,
        num_devices,
        experiment_dir,
    ):
        seed = config.get("seed", 42)
        self.model = model
        self.latent_vector_model = latent_vector_model
        self.experiment_dir = experiment_dir

        # data loading
        self.train_loader = train_loader
        self.train_iter = cycle(iter(train_loader))
        self.val_loader = val_loader

        self.trainer_config = config.train
        self.val_config = config.valid
        self.num_minibatches = self.trainer_config.get("num_minibatches", 1)

        # checkpointing
        self.checkpointing_config = config.train.checkpointing
        # use Path to create a folder for the experiment id

        #unlock the config
        self.checkpointing_config.unlock()
        self.checkpointing_config.checkpoint_dir = self.experiment_dir / Path(self.checkpointing_config.checkpoint_dir_name)
        self.checkpointing_config.lock()
        self.checkpointer = ocp.StandardCheckpointer()

        self.num_steps = int(config.train.num_steps)

        self.image_shape = (
            config.dataset.resolution,
            config.dataset.resolution,
            config.dataset.num_channels,
        )
        self.log_steps = config.train.log_steps

        self.rng = jax.random.PRNGKey(seed)

        self.num_signals_per_device = example_batch.targets.shape[0] // num_devices
        self.num_devices = num_devices
        self.interpolation_type = config.model.interpolation_type
        self.latent_spatial_dim = config.model.latent_spatial_dim

        self.inner_opt = optax.sgd(learning_rate=config.train.inner_learning_rate)
        self.inner_steps = config.train.inner_steps

        self.init(config.scheduler, self.rng, example_batch)

        self.create_forward_fn()
        self.create_loss_fn()

        self.create_train_step()

    def create_forward_fn(self):
        def _forward(field_params, latent_params, coords, target):
            return self.inner_fit(
                field_params,
                latent_params,
                coords,
                target,
            )

        self.forward = jax.pmap(_forward, axis_name="i", in_axes=(None, 0, 0, 0))

    def get_latent_vector(self, latent_params):
        return jax.pmap(
            jax.vmap(self.latent_vector_model.apply, in_axes=(0,)),
            axis_name="i",
            in_axes=(0,),
        )(
            {"params": latent_params},
        )

    def create_loss_fn(self):
        def _loss_fn(params, latent_params, batch, rng):
            field_params = params
            coords = batch.inputs
            target = batch.targets
            loss, recon, latent_params = self.inner_fit(
                field_params, latent_params, coords, target
            )
            loss = jnp.mean(loss)
            cur_mse = jnp.mean(jnp.square(recon - target), axis=(-1, -2))
            cur_psnr = (-10 * jnp.log10(cur_mse)).mean()
            cur_mse = cur_mse.mean()

            metrics = {
                "loss": loss,
                "mse": cur_mse,
                "psnr": cur_psnr,
            }
            visuals = {
                "recon": recon,
                "target": target,
            }
            return loss, (metrics, visuals)

        self.loss_fn = jax.jit(_loss_fn)

    def inner_fit(self, field_params, latent_params, coords, target):
        return jax.vmap(inner_fit, in_axes=(None, 0, None, None, None, 0, None, 0))(
            field_params,
            latent_params,
            self.model,
            self.latent_vector_model,
            self.inner_opt,
            coords,
            self.inner_steps,
            target,
        )

    def init(self, scheduler_config, rng, example_batch):
        batch = self.process_batch(example_batch)

        def get_minibatch(batch, start_idx, end_idx):
            return jax.tree_map(lambda x: x[:, start_idx:end_idx], batch)

        batch = get_minibatch(
            batch, 0, self.num_signals_per_device // self.num_minibatches
        )

        coords = batch.inputs

        latent_init_rng, field_init_rng, state_rng = jax.random.split(rng, 3)

        # initialize the latent model
        _init = jax.pmap(
            jax.vmap(self.latent_vector_model.init, in_axes=(0,)),
            axis_name="i",
            in_axes=(0,),
        )

        self.latent_params = flax.core.FrozenDict(
            _init(
                jnp.broadcast_to(
                    latent_init_rng,
                    (
                        self.num_devices,
                        self.num_signals_per_device // self.num_minibatches,
                        *latent_init_rng.shape,
                    ),
                ),
            )["params"]
        )

        # TODO: initialize with jax.eval_shape()
        latent_vector = jax.pmap(
            jax.vmap(self.latent_vector_model.apply, in_axes=(0,)),
            axis_name="i",
            in_axes=(0,),
        )(
            {"params": self.latent_params},
        )

        field_params = flax.core.FrozenDict(
            self.model.init(
                field_init_rng,
                coords[0][0],
                latent_vector[0][0],
            )["params"]
        )

        # setup the learning rate scheduler
        self.lr_schedule = build_lr_scheduler(
            scheduler_config=scheduler_config,
            num_steps=self.num_steps,
        )
        # add gradient clipping to optimizer
        outer_optimizer = optax.adam(
            learning_rate=self.lr_schedule,
        )
        if self.trainer_config.get("clip_grads", None) is not None:
            outer_optimizer = optax.chain(
                optax.clip_by_global_norm(self.trainer_config.clip_grads),
                outer_optimizer,
            )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=field_params,
            tx=outer_optimizer,
            rng=state_rng,
        )

    def compile(self):
        _ = self.train_step(
            self._params, self.state, self.coords, next(iter(self.train_loader))[0]
        )

    def validate(self, global_step):
        metrics = {
            "psnr": [],
            "mse": [],
            "psnr_batch_size": [],
            "mse_batch_size": [],
        }

        def add_metrics(new_metrics):
            for key in new_metrics.keys():
                if isinstance(new_metrics[key], jnp.ndarray):
                    metrics[key].append(new_metrics[key].sum())
                    metrics[key + "_batch_size"].append(new_metrics[key].shape[0])
                elif isinstance(new_metrics[key], list):
                    metrics[key].extend(new_metrics[key])
                    metrics[key + "_batch_size"].extend(
                        [new_metrics[key].shape[0]] * new_metrics[key].shape[0]
                    )
                elif isinstance(new_metrics[key], float):
                    metrics[key].append(new_metrics[key])
                    metrics[key + "_batch_size"].append(1)

        val_steps = len(self.val_loader)
        iter_loader = iter(self.val_loader)

        for step in tqdm(range(val_steps), desc="Validation", total=val_steps):
            batch = next(iter_loader)
            batch = self.process_batch(batch)
            coords = batch.inputs
            target = batch.targets
            recon = jax.device_get(
                self.forward(self.state.params, self.latent_params, coords, target)[1]
            )
            recon = recon.reshape(-1, *self.image_shape)
            target = target.reshape(-1, *self.image_shape)
            local_metrics = {
                "psnr": psnr(recon, target),
                "mse": jnp.mean(jnp.square(recon - target), axis=(1, 2, 3)),
            }
            add_metrics(local_metrics)
            
            if step == 0:
                plot_target = jax.device_get(target.reshape(-1, *self.image_shape))

                for i in range(min(5, self.num_signals_per_device)):
                    wandb.log(
                        {
                            f"val_images/recon_{i}": wandb.Image(recon[i]),
                            f"val_images/target_{i}": wandb.Image(plot_target[i]),
                            "val_images/recon_distribution": wandb.Histogram(
                                recon[i].flatten()
                            ),
                            "val_images/target_distribution": wandb.Histogram(
                                plot_target[i].flatten()
                            ),
                        },
                        step=global_step,
                    )
        for key in ["psnr", "mse"]:
            metrics[key] = sum(metrics[key]) / sum(metrics[key + "_batch_size"])

        wandb.log(
            {
                "val_metrics/psnr": metrics["psnr"],
                "val_metrics/mse": metrics["mse"],
            },
            step=global_step,
        )

        if self.checkpointing_config is not None:
            tracked_metric = self.checkpointing_config.get("tracked_metric", None)
            if tracked_metric is not None:
                assert (
                    tracked_metric in metrics.keys()
                ), f"Tracked metric {tracked_metric} not found in metrics, which are: {metrics.keys()}"
                if not hasattr(self, "best_metric"):
                    self.best_metric = metrics[tracked_metric]
                    self.save(checkpoint_name=f"best_{tracked_metric}")
                else:
                    direction = 1 if tracked_metric == "psnr" else -1
                    if (
                        direction * metrics[tracked_metric]
                        > direction * self.best_metric
                    ):
                        self.best_metric = metrics[tracked_metric]
                        self.save(checkpoint_name=f"best_{tracked_metric}")

    def save(self, checkpoint_name="latest"):
        if self.checkpointing_config is not None:
            checkpoint_dir = self.checkpointing_config.get("checkpoint_dir", "ckpts")
            path = (Path(f"{checkpoint_dir}") / Path(f"{checkpoint_name}")).absolute()
            self.checkpointer.save(
                path,
                self.state.params,
                ocp.args.StandardSave(self.state.params),
                force=True,
            )

    def load(self, path):
        self.state = self.state.replace(
            params=self.checkpointer.restore(
                path, ocp.args.StandardRestore(self.state.params)
            )
        )

    def train(self):
        # print all the shape of the parameters
        print(jax.tree_map(lambda x: x.shape, self.state.params))
        print(jax.tree_map(lambda x: x.shape, self.latent_params))

        for step in range(self.num_steps):
            batch = next(self.train_iter)
            batch = self.process_batch(batch)

            set_profiler(
                self.trainer_config.profiler, step, self.experiment_dir
            )

            if step % self.val_config.val_interval == 0 or (step == self.num_steps - 1):
                self.validate(step)

            self.state, metrics, visuals = self.train_step(
                self.latent_params, self.state, batch
            )

            if self.checkpointing_config is not None and (
                step % self.checkpointing_config.checkpoint_interval == 0
                or (step == self.num_steps - 1)
            ):
                self.save()

            if step % self.log_steps.loss == 0 or (step == self.num_steps - 1):
                # log the learning rate
                learning_rate = extract_learning_rate(
                    self.lr_schedule, self.state.opt_state
                )
                wandb.log(
                    {
                        "train_metrics/loss": metrics["loss"],
                        "train_metrics/mse": metrics["mse"],
                        "train_metrics/psnr": metrics["psnr"],
                        "train_metrics/learning_rate": learning_rate,
                    },
                    step=step,
                )
                logging.info(
                    f"Step {step} | Loss {metrics['loss']} | Learning rate {learning_rate}"
                )
                if "lrs" in self.state.params.keys():
                    internal_learning_rate = self.state.params["lrs"]["lrs"]

                    wandb.log(
                        {
                            "train_metrics/internal_learning_rate": wandb.Histogram(
                                internal_learning_rate
                            ),
                        },
                        step=step,
                    )

            if step % self.log_steps.image == 0 or (step == self.num_steps - 1):
                # evaluate the mdoel
                recon = jax.device_get(
                    visuals["recon"][0].reshape(-1, *self.image_shape)
                )
                plot_target = jax.device_get(
                    batch.targets.reshape(-1, *self.image_shape)
                )

                for i in range(min(5, self.num_signals_per_device)):
                    # get from device jax
                    wandb.log(
                        {
                            f"train_images/recon_{i}": wandb.Image(recon[i]),
                            f"train_images/target_{i}": wandb.Image(plot_target[i]),
                        },
                        step=step,
                    )
                    # log the image statistics
                    wandb.log(
                        {
                            "train_images/recon_distribution": wandb.Histogram(
                                recon[i].flatten()
                            ),
                            "train_images/target_distribution": wandb.Histogram(
                                plot_target[i].flatten()
                            ),
                        },
                        step=step,
                    )

    def process_batch(self, batch: Batch):
        # extract necessary information from inputs (coords) and target (image)
        batch_size = batch.inputs.shape[0]
        num_signals_per_device = batch_size // self.num_devices
        num_channels = batch.targets.shape[-1]

        # reshape to account for the number of devices
        inputs = batch.inputs.reshape(self.num_devices, num_signals_per_device, -1, 2)
        targets = batch.targets.reshape(
            self.num_devices, num_signals_per_device, -1, num_channels
        )
        labels = batch.labels.reshape(
            self.num_devices,
            num_signals_per_device,
        )
        return dataclasses.replace(batch, inputs=inputs, targets=targets, labels=labels)

    def create_train_step(self):
        def _train_step(latent_params, state, batch):
            rng, cur_rng = jax.random.split(state.rng)
            _loss_fn = lambda params, batch, rng: self.loss_fn(
                params, latent_params, batch, rng
            )
            grads, metrics, visuals = accumulate_gradients(
                state.params,
                batch,
                cur_rng,
                self.trainer_config.get("num_minibatches", 1),
                _loss_fn,
            )
            # grads = jax.tree_util.tree_map(
            #     lambda x: x / self.num_signals_per_device, grads
            # )
            grads = jax.lax.psum(grads, axis_name="i")

            metrics = jax.lax.pmean(metrics, axis_name="i")

            state = state.apply_gradients(grads=grads, rng=rng)
            return state, metrics, visuals

        self.train_step = jax.pmap(
            _train_step,
            axis_name="i",
            in_axes=(0, None, 0),
            out_axes=(None, None, 0),
        )
