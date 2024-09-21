import json
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2

import git
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
from absl import logging
from ml_collections import config_dict, config_flags
from ml_collections.config_dict import ConfigDict
from torchvision import transforms

from spatial_functa.grad_acc import Batch

DATA_PATH = os.environ.get("DATA_PATH", "data")


def get_coords(height, width, center_pixel=True, zero_one=True):
    if zero_one:
        starting_coord = 0
        ending_coord = 1
    else:
        starting_coord = -1
        ending_coord = 1

    if center_pixel:
        half_pixel = (ending_coord - starting_coord) / (
            2 * height
        )  # Size of half a pixel in grid
        x = np.linspace(starting_coord + half_pixel, ending_coord - half_pixel, height)
        half_pixel = (ending_coord - starting_coord) / (
            2 * width
        )  # Size of half a pixel in grid
        y = np.linspace(starting_coord + half_pixel, ending_coord - half_pixel, width)
    else:
        x = np.linspace(starting_coord, ending_coord, height)
        y = np.linspace(starting_coord, ending_coord, width)
    x, y = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([x, y], axis=-1)
    return coords


def image_to_numpy(image):
    image = np.array(image, dtype=np.float32) / 255.0
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def normalize(image, mean: np.array, std: np.array):
    return (image - mean) / std


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return [numpy_collate(samples) for samples in zip(*batch)]
    else:
        return np.array(batch)


def get_means_std(dataset_name: str):
    if dataset_name == "cifar10":
        DatasetClass = torchvision.datasets.CIFAR10
    elif dataset_name == "mnist":
        DatasetClass = torchvision.datasets.MNIST
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = DatasetClass(DATA_PATH, train=True, download=True)

    DATA_MEANS = (dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (dataset.data / 255.0).std(axis=(0, 1, 2))

    return DATA_MEANS, DATA_STD


def random_h_flip(image, rng: np.random.RandomState):
    if rng.random() < 0.5:
        return image
    return np.fliplr(image)


def scale_and_random_crop(
    image: np.array,
    rng: np.random.RandomState,
    resize_w: int,
    resize_h: int,
    final_w: int,
    final_h: int,
):
    # image_h, image_w = image.shape[:2]

    scaled_image = cv2.resize(
        image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
    )

    # random crop
    crop_x = rng.randint(0, resize_w - final_w)
    crop_y = rng.randint(0, resize_h - final_h)

    return scaled_image[crop_y : crop_y + final_h, crop_x : crop_x + final_w]


class ScaleAndRandomCrop:
    def __init__(self, pad_w, pad_h, crop_w, crop_h, seed):
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.rng = np.random.RandomState(seed)

    def __call__(self, image):
        return scale_and_random_crop(
            image, self.rng, self.pad_w, self.pad_h, self.crop_w, self.crop_h
        )


class RandomHorizontalFlip:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, image):
        return random_h_flip(image, self.rng)


def get_augmented_dataloader(
    dataset_config: ConfigDict,
    subset: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_minibatches: int = 1,
):
    dataset_name = dataset_config.name
    resolution = dataset_config.resolution
    if dataset_name == "cifar10":
        DatasetClass = torchvision.datasets.CIFAR10
        train_size = 45000
        val_size = 5000
    elif dataset_name == "mnist":
        DatasetClass = torchvision.datasets.MNIST
        train_size = 50000
        val_size = 10000
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    num_augmentations = int(dataset_config.get("num_augmentations", 1))

    if subset == "train":

        def make_transform(cur_seed):
            if dataset_config.get("apply_augment", False):
                resolution = dataset_config.resolution
                resize_resolution = int(resolution * 1.25)
                return transforms.Compose(
                    [
                        image_to_numpy,
                        RandomHorizontalFlip(cur_seed),
                        ScaleAndRandomCrop(
                            resize_resolution,
                            resize_resolution,
                            resolution,
                            resolution,
                            cur_seed,
                        ),
                    ]
                )
            else:
                return transforms.Compose(
                    [
                        image_to_numpy,
                    ]
                )

    else:
        data_transform = transforms.Compose(
            [
                image_to_numpy,
            ]
        )

    if subset == "train":
        dataset_list = []
        for i in range(num_augmentations):
            cur_seed = seed + i
            dataset = DatasetClass(
                DATA_PATH, train=True, download=True, transform=make_transform(cur_seed)
            )
            dataset_list.append(
                torch.utils.data.Subset(dataset, list(range(train_size)))
            )
        dataset = torch.utils.data.ConcatDataset(dataset_list)
    elif subset == "val":
        dataset = DatasetClass(
            DATA_PATH, train=True, download=True, transform=data_transform
        )
        dataset = torch.utils.data.Subset(
            dataset, list(range(train_size, train_size + val_size))
        )
    elif subset == "test":
        dataset = DatasetClass(
            DATA_PATH, train=False, download=True, transform=data_transform
        )

    logging.info(f"Using {len(dataset)} samples for {subset}")
    coords = get_coords(
        resolution,
        resolution,
        center_pixel=dataset_config.get("center_pixel", True),
        zero_one=dataset_config.get("zero_one", True),
    )

    batch_size = batch_size * jax.device_count()
    if subset != "train":
        batch_size = batch_size // num_minibatches

    coords = np.repeat(coords[None, ...], batch_size, axis=0)

    def batch_collate(batch):
        batch_list = numpy_collate(batch)
        return Batch(inputs=coords, targets=batch_list[0], labels=batch_list[1])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=dataset_config.get("num_workers", 12),
    )
    return dataloader
