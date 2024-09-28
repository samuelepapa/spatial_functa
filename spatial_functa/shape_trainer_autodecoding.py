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

from copy import deepcopy

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


def forward(params, model, latent_model, coords, idx):
    latent_params = params["latent_params"]
    field_params = params["field_params"]
    latent_vector = latent_model.apply({"params": latent_params}, idx)
    pred = model.apply({"params": field_params}, coords, latent_vector)
    return pred[0]


mse_fn = jax.jit(lambda x, y: jnp.mean(jnp.square(x - y)))


def iou(pred, target):
    pred = jnp.where(pred < 0, 1, 0)
    target = jnp.where(target < 0, 1, 0)
    intersection = jnp.sum(pred * target, axis=(-1, -2))
    union = jnp.sum(pred, axis=(-1, -2)) + jnp.sum(target, axis=(-1, -2)) - intersection
    return intersection / union


def loss_fn_image(params, model, latent_model, coords, target, idx):
    pred = forward(params, model, latent_model, coords, idx)
    return mse_fn(pred, target), pred


@jax.named_scope("inner_fit")
def inner_fit(
    params,
    model,
    latent_model,
    coords,
    target,
    idx,
) -> Tuple[Array, Array, Array]:
    # final loss computation
    loss, recon = loss_fn_image(params, model, latent_model, coords, target, idx)

    return loss, recon


class ShapeTrainer:
    def __init__(
        self,
        model,
        lr_model,
        latent_vector_model,
        latent_vector_model_val,
        example_batch,
        config,
        train_loader,
        val_loader,
        metrics_loader,
        num_devices,
    ):
        seed = config.get("seed", 42)
        self.model = model
        self.latent_vector_model = latent_vector_model
        self.latent_vector_model_val = latent_vector_model_val
        self.lr_model = lr_model
        self.inner_lr_scaling = config.train.inner_lr_scaling
        self.coords_dim = config.dataset.coords_dim

        # data loading
        self.train_loader = train_loader
        self.train_iter = cycle(iter(train_loader))
        self.val_loader = val_loader
        self.metrics_loader = metrics_loader

        self.trainer_config = config.train
        self.val_config = config.valid
        self.num_minibatches = self.trainer_config.get("num_minibatches", 1)

        # number of points to sample per signal
        self.num_points = config.dataset.resolution
        # create a loop of random indices
        total_points = config.dataset.total_points
        # create 100 permutations of the indices
        num_permutations = 100 * config.train.inner_steps
        permutations = np.concatenate(
            [np.random.permutation(total_points) for _ in range(num_permutations)]
        )
        available_points = total_points * num_permutations
        available_points = available_points - (available_points % self.num_points)

        # reshape the permutations to have indices for each signal
        self.indices = permutations[:available_points].reshape(
            -1, config.train.inner_steps, self.num_points
        )
        self.batch_idx = 0
        self.num_batches = self.indices.shape[0]
        # checkpointing
        self.checkpointing_config = config.train.checkpointing
        self.checkpointer = ocp.StandardCheckpointer()

        self.num_steps = int(config.train.num_steps)

        self.sdf_shape = (
            config.dataset.resolution,
            config.dataset.num_channels,
        )
        self.log_steps = config.train.log_steps

        self.rng = jax.random.PRNGKey(seed)

        self.num_points_per_device = example_batch.targets.shape[0] // num_devices
        self.num_devices = num_devices
        self.interpolation_type = config.model.interpolation_type
        self.latent_spatial_dim = config.model.latent_spatial_dim

        self.inner_opt = optax.sgd(learning_rate=config.train.inner_learning_rate)
        if self.trainer_config.get("clip_grads", None) is not None:
            self.inner_opt = optax.chain(
                optax.clip_by_global_norm(self.trainer_config.clip_grads),
                self.inner_opt,
            )
        self.inner_steps = config.train.inner_steps

        self.init(config.scheduler, self.rng, example_batch)

        self.create_forward_fn_train()
        self.create_forward_fn_val()
        self.create_loss_fn()
        self.create_val_loss_fn()

        self.create_train_step()
        self.create_val_step()

    def create_forward_fn_train(self):
        def _forward(params, coords, target, idx):
            return self.inner_fit(
                params,
                coords,
                target,
                idx,
                self.latent_vector_model,
            )

        self.forward_train = jax.pmap(_forward, axis_name="i", in_axes=(None, 0, 0, 0))

    def create_forward_fn_val(self):
        def _forward(params, coords, target, idx):
            return self.inner_fit(
                params,
                coords,
                target,
                idx,
                self.latent_vector_model_val,
            )

        self.forward_val = jax.pmap(_forward, axis_name="i", in_axes=(None, 0, 0, 0))


    def get_latent_vector(self, latent_params):
        return jax.pmap(
            jax.vmap(self.latent_vector_model.apply, in_axes=(0,)),
            axis_name="i",
            in_axes=(0,),
        )(
            {"params": latent_params},
        )

    def create_loss_fn(self):
        def _loss_fn(params, batch, rng):
            coords = batch.inputs
            target = batch.targets
            idx = batch.signal_idxs
            loss, recon = self.inner_fit(params, coords, target, idx, self.latent_vector_model)
            loss = jnp.mean(loss)
            cur_mse = jnp.mean(jnp.square(recon - target[:, -1]), axis=(-1, -2))
            cur_mse = cur_mse.mean()

            metrics = {
                "loss": loss,
                "mse": cur_mse,
            }
            visuals = {
                "recon": recon,
                "target": target[-1],
            }
            return loss, (metrics, visuals)

        self.loss_fn = jax.jit(_loss_fn)
        
    def create_val_loss_fn(self):
        def _val_loss_fn(params, batch, rng):
            coords = batch.inputs
            target = batch.targets
            idx = batch.signal_idxs
            loss, recon = self.inner_fit(params, coords, target, idx, self.latent_vector_model_val)
            loss = jnp.mean(loss)
            cur_mse = jnp.mean(jnp.square(recon - target[:, -1]), axis=(-1, -2))
            cur_mse = cur_mse.mean()

            metrics = {
                "loss": loss,
                "mse": cur_mse,
            }
            visuals = {
                "recon": recon,
                "target": target[-1],
            }
            return loss, (metrics, visuals)

        self.val_loss_fn = jax.jit(_val_loss_fn)

    def inner_fit(self, field_params, coords, target, idx, latent_vector_model):
        return jax.vmap(
            inner_fit, in_axes=(None, None, None, 0, 0, 0)
        )(
            field_params,
            self.model,
            latent_vector_model,
            coords,
            target,
            idx,
        )

    def init(self, scheduler_config, rng, example_batch):
        batch = self.process_batch(example_batch)

        def get_minibatch(batch, start_idx, end_idx):
            return jax.tree_map(lambda x: x[:, start_idx:end_idx], batch)

        batch = get_minibatch(
            batch, 0, self.num_points_per_device // self.num_minibatches
        )

        coords = batch.inputs

        latent_init_rng, field_init_rng, state_rng = jax.random.split(rng, 3)

        # initialize the latent model
        latent_params = flax.core.FrozenDict(
            self.latent_vector_model.init(
                latent_init_rng,
                batch.signal_idxs[0][0],
            )["params"]
        )

        # TODO: initialize with jax.eval_shape()
        latent_vector = jax.pmap(
            jax.vmap(self.latent_vector_model.apply, in_axes=(None, 0)),
            axis_name="i",
            in_axes=(None, 0),
        )({"params": latent_params}, batch.signal_idxs)

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
        if self.trainer_config.get("outer_clip_grads", None) is not None:
            outer_optimizer = optax.chain(
                optax.clip_by_global_norm(self.trainer_config.outer_clip_grads),
                outer_optimizer,
            )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params={
                "field_params": field_params,
                "latent_params": latent_params,
            },
            tx=outer_optimizer,
            rng=state_rng,
        )

    def compile(self):
        _ = self.train_step(
            self._params, self.state, self.coords, next(iter(self.train_loader))[0]
        )
        
    def compute_metrics(self, global_step, state):
        metrics = {
            "mse": [],
            "mse_batch_size": [],
            "iou": [],
            "iou_batch_size": [],
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
        
        iter_loader = iter(self.metrics_loader)
        num_steps = len(self.metrics_loader)
        
        for step in tqdm(range(num_steps), desc="Metrics", total=num_steps):
            batch = next(iter_loader)
            batch = self.process_batch(batch)
            target = batch.targets
            loss, recon = self.forward_val(
                state.params,
                batch.inputs,
                batch.targets,
                batch.signal_idxs,
            )
            cur_iou = iou(recon, target)
            cur_mse = jnp.mean(jnp.square(recon - target), axis=(1, 2))
            add_metrics({"iou": cur_iou, "mse": cur_mse})
            
        for key in ["mse", "iou"]:
            metrics[key] = sum(metrics[key]) / sum(metrics[key + "_batch_size"])
            logging.info(f"Metrics | {key} {metrics[key]}")
        wandb.log(
            {
                "metrics/mse": metrics["mse"],
                "metrics/iou": metrics["iou"],
            },
            step=global_step,
        )
            
        return metrics

    def validate(self, global_step):
        example_batch = self.process_batch(next(iter(self.val_loader)))
        
        iter_loader = cycle(iter(self.val_loader))

        val_optim = optax.adam(learning_rate=self.lr_schedule)
        # re-init the latent params for the validation dataset
        latent_init_rng, state_rng = jax.random.split(self.rng, 2)
        # initialize the latent model
        latent_params = flax.core.FrozenDict(
            self.latent_vector_model_val.init(
                latent_init_rng,
                example_batch.signal_idxs,
            )["params"]
        )
        # make a copy of the params
        params = deepcopy(self.state.params)
        params['latent_params'] = latent_params
        valid_state = self.state.replace(
            opt_state=val_optim.init(params), params=params
        )
        
        log_steps = self.log_steps.loss
        
        val_steps = self.val_config.num_epochs * len(self.val_loader)
        for step in tqdm(range(val_steps), desc="Validation", total=val_steps):
            batch = next(iter_loader)
            batch = self.process_batch(batch)
            valid_state, cur_metrics, visuals = self.val_step(valid_state, batch)
            
            if step % log_steps == 0:
                wandb.log(
                    {
                        "val_metrics/mse": cur_metrics["mse"].mean().item(),
                    },
                    step=global_step,
                )
            global_step += 1

        metrics = self.compute_metrics(global_step, valid_state)         

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

        return global_step

    def save(self, checkpoint_name="latest"):
        if self.checkpointing_config is not None:
            checkpoint_dir = self.checkpointing_config.get("checkpoint_dir", "ckpts")
            path = (Path(f"{checkpoint_dir}") / Path(f"{checkpoint_name}")).absolute()
            self.checkpointer.save(
                path,
                self.state.params,
                # ocp.args.StandardSave(self.state.params), removed due to orbax update
                force=True,
            )

    def load(self, path):
        self.state = self.state.replace(
            params=self.checkpointer.restore(path, self.state.params)
        )

    def train(self):
        # print all the shape of the parameters
        logging.info(jax.tree_map(lambda x: x.shape, self.state.params))
        
        global_step = 0

        for _ in range(self.num_steps):
            batch = next(self.train_iter)
            batch = self.process_batch(batch)

            set_profiler(
                self.trainer_config.profiler, global_step, self.trainer_config.log_dir
            )

            if global_step % self.val_config.val_interval == 0 or (global_step == self.num_steps - 1):
                global_step = self.validate(global_step)

            self.state, metrics, visuals = self.train_step(self.state, batch)

            if self.checkpointing_config is not None and (
                global_step % self.checkpointing_config.checkpoint_interval == 0
                or (global_step == self.num_steps - 1)
            ):
                self.save()

            if global_step % self.log_steps.loss == 0 or (global_step == self.num_steps - 1):
                # log the learning rate
                learning_rate = extract_learning_rate(
                    self.lr_schedule, self.state.opt_state
                )
                wandb.log(
                    {
                        "train_metrics/loss": metrics["loss"],
                        "train_metrics/mse": metrics["mse"],
                        "train_metrics/learning_rate": learning_rate,
                    },
                    step=global_step,
                )
                logging.info(
                    f"Step {global_step} | Loss {metrics['loss']} | Learning rate {learning_rate}"
                )
                if "lrs" in self.state.params.keys():
                    internal_learning_rate = self.state.params["lrs"]["lrs"]

                    wandb.log(
                        {
                            "train_metrics/internal_learning_rate": wandb.Histogram(
                                internal_learning_rate
                            ),
                        },
                        step=global_step,
                    )
            global_step += 1

    def process_batch(self, batch: Batch):
        # extract necessary information from inputs (coords) and target (image)
        batch_size = batch.inputs.shape[0]
        if batch_size == 1:
            batch_size = batch.inputs.shape[1]
            batch = dataclasses.replace(
                batch,
                labels=jnp.broadcast_to(batch.labels, (batch_size, *batch.labels.shape)),
                signal_idxs=jnp.broadcast_to(
                    batch.signal_idxs, (batch_size, *batch.signal_idxs.shape)
                ),
            )
        num_points_per_device = batch_size // self.num_devices
        assert (
            num_points_per_device % self.num_minibatches == 0
        ), f"Batch size {num_points_per_device} is not divisible by num_minibatches {self.num_minibatches}"
        num_channels = batch.targets.shape[-1]

        # reshape to account for the number of devices
        inputs = batch.inputs.reshape(
            self.num_devices, num_points_per_device, self.coords_dim
        )
        targets = batch.targets.reshape(
            self.num_devices, num_points_per_device, num_channels
        )
        labels = batch.labels.reshape(
            self.num_devices,
            num_points_per_device,
        )
        signal_idxs = batch.signal_idxs.reshape(
            self.num_devices,
            num_points_per_device,
        )

        return dataclasses.replace(
            batch,
            inputs=inputs,
            targets=targets,
            labels=labels,
            signal_idxs=signal_idxs,
        )

    def create_train_step(self):
        def _train_step(state, batch):
            rng, cur_rng = jax.random.split(state.rng)
            _loss_fn = lambda params, batch, rng: self.loss_fn(params, batch, rng)
            grads, metrics, visuals = accumulate_gradients(
                state.params,
                batch,
                cur_rng,
                self.trainer_config.get("num_minibatches", 1),
                _loss_fn,
            )
            # grads = jax.tree_util.tree_map(
            #     lambda x: x / self.num_points_per_device, grads
            # )
            grads = jax.lax.pmean(grads, axis_name="i")

            metrics = jax.lax.pmean(metrics, axis_name="i")

            state = state.apply_gradients(grads=grads, rng=rng)
            return state, metrics, visuals

        self.train_step = jax.pmap(
            _train_step,
            axis_name="i",
            in_axes=(None, 0),
            out_axes=(None, None, 0),
        )
        
    def create_val_step(self):
        def _val_step(state, batch):
            rng, cur_rng = jax.random.split(state.rng)
            _loss_fn = lambda params, batch, rng: self.val_loss_fn(params, batch, rng)
            grads, metrics, visuals = accumulate_gradients(
                state.params,
                batch,
                cur_rng,
                self.trainer_config.get("num_minibatches", 1),
                _loss_fn,
            )
            # grads = jax.tree_util.tree_map(
            #     lambda x: x / self.num_points_per_device, grads
            # )
            grads = jax.lax.pmean(grads, axis_name="i")

            metrics = jax.lax.pmean(metrics, axis_name="i")

            state = state.apply_gradients(grads=grads, rng=rng)
            return state, metrics, visuals
        
        self.val_step = jax.pmap(
            _val_step,
            axis_name="i",
            in_axes=(None, 0),
            out_axes=(None, None, 0),
        )
