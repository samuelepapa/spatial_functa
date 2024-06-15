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


def forward(params, model, latent_vector, rng, train):
    pred = model.apply(
        {"params": params}, latent_vector, train=train, rngs={"dropout": rng}
    )
    return pred

class Trainer:
    def __init__(
        self,
        model,
        example_batch,
        config,
        train_loader,
        val_loader,
        num_devices,
    ):
        seed = config.get("seed", 42)
        self.model = model

        # data loading
        self.train_loader = train_loader
        self.train_iter = cycle(iter(train_loader))
        self.val_loader = val_loader

        self.trainer_config = config.train
        self.val_config = config.valid
        self.model_config = config.model
        self.num_minibatches = self.trainer_config.get("num_minibatches", 1)

        # checkpointing
        self.checkpointing_config = config.train.checkpointing
        self.checkpointer = ocp.StandardCheckpointer()

        self.num_steps = int(config.train.num_steps)

        self.image_shape = (
            config.dataset.resolution,
            config.dataset.resolution,
            config.dataset.num_channels,
        )
        self.log_steps = config.train.log_steps

        self.rng = jax.random.PRNGKey(seed)

        self.num_signals_per_device = example_batch.labels.shape[0] // num_devices
        self.num_devices = num_devices

        self.init(config.scheduler, self.rng, example_batch)

        self.create_forward_fn()
        self.create_loss_fn()

        self.create_train_step()

    def create_forward_fn(self):
        def _forward(params, latent_vector, rng, train):
            latent_vector = latent_vector / self.trainer_config.normalizing_factor
            rngs = jax.random.split(rng, latent_vector.shape[0])
            return jax.vmap(forward, in_axes=(None, None, 0, 0, None))(
                params, self.model, latent_vector, rngs, train
            )

        self.forward = jax.jit(_forward)
        self.pmapped_forward = jax.pmap(_forward, axis_name="i", in_axes=(None, 0, 0, None))

    def create_loss_fn(self):
        def _loss_fn(params, batch, rng):
            field_params = params
            latent_vector = batch.inputs
            labels = batch.labels
            logits = self.forward(field_params, latent_vector, rng=rng, train=True)

            loss = jnp.mean(
                optax.softmax_cross_entropy(logits=logits, labels=labels)
            )
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1))

            loss = jnp.mean(loss)

            metrics = {
                "loss": loss,
                "acc": acc,
            }
            visuals = {}
            return loss, (metrics, visuals)

        self.loss_fn = jax.jit(_loss_fn)

    def init(self, scheduler_config, rng, example_batch):
        batch = self.process_batch(example_batch)

        def get_minibatch(batch, start_idx, end_idx):
            return jax.tree_map(lambda x: x[:, start_idx:end_idx], batch)

        batch = get_minibatch(
            batch, 0, self.num_signals_per_device // self.num_minibatches
        )

        latent_vector = batch.inputs[0][0]

        classifier_init_rng, state_rng = jax.random.split(rng, 2)

        params = flax.core.FrozenDict(
            self.model.init(
                classifier_init_rng,
                latent_vector,
            )["params"]
        )

        # setup the learning rate scheduler
        self.lr_schedule = build_lr_scheduler(
            scheduler_config=scheduler_config,
            num_steps=self.num_steps,
        )
        # add gradient clipping to optimizer
        optimizer = optax.adamw(
            learning_rate=self.lr_schedule,
            weight_decay=self.trainer_config.weight_decay,
        )
        if self.trainer_config.get("clip_grads", None) is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.trainer_config.clip_grads),
                optimizer,
            )

        new_rngs = jax.random.split(state_rng, self.num_devices + 1)
        self.per_device_rng = new_rngs[:-1]

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            rng=new_rngs[-1],
        )

    def validate(self, global_step):
        metrics = {
            "loss": [],
            "acc": [],
            "loss_batch_size": [],
            "acc_batch_size": [],
        }

        def add_metrics(new_metrics):
            for key in new_metrics.keys():
                if isinstance(new_metrics[key], (jnp.ndarray, np.ndarray)):
                    if new_metrics[key].ndim == 0:
                        metrics[key].append(new_metrics[key])
                        metrics[key + "_batch_size"].append(1)
                    else:
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
                else:
                    raise ValueError(
                        f"Unsupported type for metric {key}: {type(new_metrics[key])}"
                    )

        val_steps = len(self.val_loader)
        iter_loader = iter(self.val_loader)

        cur_rng = self.per_device_rng[0]

        same_rng_per_device = jnp.broadcast_to(cur_rng, (self.num_devices,) + cur_rng.shape)

        for step in tqdm(range(val_steps), desc="Validation", total=val_steps):
            batch = next(iter_loader)
            batch = self.process_batch(batch)
            latent_vector = batch.inputs
            labels = batch.labels
            logits = jax.device_get(
                self.pmapped_forward(self.state.params, latent_vector, same_rng_per_device, False)
            )
            loss = jnp.mean(
                optax.softmax_cross_entropy(logits=logits, labels=labels)
            )
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1))

            local_metrics = {
                "loss": jax.device_get(loss),
                "acc": jax.device_get(acc),
            }
            add_metrics(local_metrics)
            

        for key in ["loss", "acc"]:
            metrics[key] = sum(metrics[key]) / sum(metrics[key + "_batch_size"])

        wandb.log(
            {
                "val_metrics/loss": metrics["loss"],
                "val_metrics/acc": metrics["acc"],
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
                    direction = 1 if tracked_metric == "acc" else -1
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

    def update_per_device_rng(self):
        self.state = self.state.replace(
            rng=jax.random.split(self.state.rng, self.num_devices)
        )

    def train(self):
        # print all the shape of the parameters
        print(jax.tree_map(lambda x: x.shape, self.state.params))

        for step in range(self.num_steps):
            batch = next(self.train_iter)
            batch = self.process_batch(batch)

            set_profiler(
                self.trainer_config.profiler, step, self.trainer_config.log_dir
            )

            if step % self.val_config.val_interval == 0 or (step == self.num_steps - 1):
                self.validate(step)

            self.state, metrics, _, self.per_device_rng = self.train_step(
                self.state, batch, self.per_device_rng
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
                        "train_metrics/acc": metrics["acc"],
                        "train_metrics/learning_rate": learning_rate,
                    },
                    step=step,
                )
                logging.info(
                    f"Step {step} | Loss {metrics['loss']} | Learning rate {learning_rate} | Acc {metrics['acc']}"
                )

    def process_batch(self, batch: Batch):
        # extract necessary information from inputs (coords) and target (image)
        batch_size = batch.inputs.shape[0]
        num_signals_per_device = batch_size // self.num_devices

        # reshape to account for the number of devices
        if self.model_config.name == "mlp":
            inputs = batch.inputs.reshape(self.num_devices, num_signals_per_device, -1)
        elif self.model_config.name == "transformer":
            inputs = batch.inputs.reshape(
                self.num_devices, num_signals_per_device, self.model_config.num_patches, -1
            )
        else:
            raise ValueError(f"Unknown model {self.model_config.name}")
        
        labels = batch.labels.reshape(
            self.num_devices,
            num_signals_per_device,
        )
        # one-hot encode the labels
        num_classes = self.trainer_config.num_classes
        labels = jax.nn.one_hot(labels, num_classes)
        if self.trainer_config.label_smoothing == True:
            l = self.trainer_config.label_smoothing_factor
            labels = labels * (1 - l) + l / num_classes

        return dataclasses.replace(batch, inputs=inputs, labels=labels)

    def create_train_step(self):
        def _train_step(state, batch, per_device_rng):
            _loss_fn = lambda params, batch, rng: self.loss_fn(
                params, batch, rng
            )

            per_device_rng, cur_rng = jax.random.split(per_device_rng)
            grads, metrics, visuals = accumulate_gradients(
                state.params,
                batch,
                cur_rng,
                self.trainer_config.get("num_minibatches", 1),
                _loss_fn,
            )
            grads = jax.tree_util.tree_map(
                lambda x: x / self.num_signals_per_device, grads
            )
            grads = jax.lax.pmean(grads, axis_name="i")

            metrics = jax.lax.pmean(metrics, axis_name="i")

            state = state.apply_gradients(grads=grads)
            return state, metrics, visuals, per_device_rng

        self.train_step = jax.pmap(
            _train_step,
            axis_name="i",
            in_axes=(None, 0, 0),
            out_axes=(None, None, 0, 0),
        )
