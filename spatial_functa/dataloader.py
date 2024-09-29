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

import h5py

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
    
    if len(scaled_image.shape) == 2:
        scaled_image = np.expand_dims(scaled_image, axis=-1)

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

def get_image_dataloader(
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

def get_augmented_dataloader(
    dataset_config: ConfigDict,
    subset: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_minibatches: int = 1,
):
    dataset_name = dataset_config.name
    if dataset_name in ["cifar10", "mnist"]:
        return get_image_dataloader(
            dataset_config, subset, batch_size, shuffle, seed, num_minibatches
        )
    elif dataset_name in ["shapenet", "shapenet_batched", "shapenet_chunked"]:
        return get_shapenet_dataloader(
            dataset_config, subset, batch_size, shuffle, seed, num_minibatches
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, only cifar10, mnist, shapenet, shapenet_batched supported")

def get_shapenet_dataloader(
    dataset_config: ConfigDict,
    subset: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_minibatches: int = 1,
):
    dataset_name = dataset_config.name
    if dataset_name == "shapenet":
        DatasetClass = ShapeNetSDFH5
    elif dataset_name == "shapenet_batched":
        DatasetClass = ShapeNetSDFH5Batched
    elif dataset_name == "shapenet_chunked":
        DatasetClass = ShapeNetSDFH5Chunked
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    if dataset_name in ["shapenet", "shapenet_batched"]:
        def batch_collate(batch):
            batch_list = numpy_collate(batch)
            batch_list[1] = np.expand_dims(batch_list[1], axis=-1)
            batch_list[3] = np.expand_dims(batch_list[3], axis=-1)
            return Batch(inputs=batch_list[0], targets=batch_list[1], labels=batch_list[2], signal_idxs=batch_list[3])
    elif dataset_name == "shapenet_chunked":
        def batch_collate(batch):
            batch_list = numpy_collate(batch)
            targets = np.expand_dims(batch_list[1], axis=-1)
            targets = targets.reshape(-1, 1)
            inputs = batch_list[0].reshape(-1, 3)
            labels = np.repeat(batch_list[2], batch_list[0].shape[1], axis=0)
            signal_idxs = np.repeat(batch_list[3], batch_list[0].shape[1], axis=0)
            return Batch(inputs=inputs, targets=targets, labels=labels, signal_idxs=signal_idxs)
    
    if subset == "train":
        dataset = DatasetClass(
            DATA_PATH,
            seed=seed,
            debug=dataset_config.get("debug", False),
            train=True,
        )
    elif subset == "val":
        dataset = DatasetClass(
            DATA_PATH,
            seed=seed,
            debug=dataset_config.get("debug", False),
            train=False,
        )
    elif subset == "test":
        dataset = DatasetClass(
            DATA_PATH,
            seed=seed,
            debug=dataset_config.get("debug", False),
            train=False,
        )

    logging.info(f"Using {len(dataset)} samples for {subset}")

    batch_size = batch_size * jax.device_count()
    if subset != "train":
        batch_size = batch_size // num_minibatches


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=dataset_config.get("num_workers", 12),
    )
    return dataloader

class ShapeNetSDFH5(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        seed: int = 42,
        debug: bool = False,
        train: bool = True,
    ):
        self.root = root
        self.debug = debug
        if train:
            self.file_path = os.path.join(self.root, "shapenet_train.h5")
        else:
            self.file_path = os.path.join(self.root, "shapenet_test.h5")

        self.data = h5py.File(self.file_path, "r")
        self.num_signals = self.data["indices"].shape[0]
        self.points_per_signal = self.data["points"].shape[0] // self.num_signals
        self.total_num_points = self.data["points"].shape[0]
        if self.debug:
            if train:
                self.num_signals = 2056
            else:
                self.num_signals = 1000
            self.points = self.data["points"][:self.num_signals*self.points_per_signal]
            self.sdf = self.data["sdf"][:self.num_signals*self.points_per_signal]
        else:
            self.points = self.data["points"][:]
            self.sdf = self.data["sdf"][:]
        self.points = self.points.astype(np.float32).reshape(
            self.num_signals, self.points_per_signal, 3
        )
        self.sdf = self.sdf.astype(np.float32).reshape(
            self.num_signals, self.points_per_signal
        )

        self.labels = self.data["labels"][:]

    def __len__(self):
        return self.num_signals

    def __getitem__(self, idx):
        points = self.points[idx]
        sdf = self.sdf[idx]

        # # Get label
        label = self.labels[idx]
        return points, sdf, label, idx
    
class ShapeNetSDFH5Batched(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        seed: int = 42,
        debug: bool = False,
        train: bool = True,
    ):
        self.root = root
        self.debug = debug
        if train:
            self.file_path = os.path.join(self.root, "shapenet_train.h5")
        else:
            self.file_path = os.path.join(self.root, "shapenet_test.h5")

        self.data = h5py.File(self.file_path, "r")
        self.num_signals = self.data["indices"].shape[0]
        self.points_per_signal = self.data["points"].shape[0] // self.num_signals
        self.total_num_points = self.data["points"].shape[0]
        if self.debug:
            if train:
                self.num_signals = 256
            else:
                self.num_signals = 64
            self.points = self.data["points"][:self.num_signals*self.points_per_signal]
            self.sdf = self.data["sdf"][:self.num_signals*self.points_per_signal]
            self.total_num_points = self.num_signals*self.points_per_signal
        else:
            self.points = self.data["points"][:]
            self.sdf = self.data["sdf"][:]
        # self.points = self.points.astype(np.float32).reshape(
        #     self.num_signals, self.points_per_signal, 3
        # )
        # self.sdf = self.sdf.astype(np.float32).reshape(
        #     self.num_signals, self.points_per_signal
        # )

        self.labels = self.data["labels"][:]

    def __len__(self):
        return self.total_num_points

    def __getitem__(self, idx):
        signal_idx, point_idx = divmod(idx, self.points_per_signal)
        points = self.points[idx]
        sdf = self.sdf[idx]

        # # Get label
        label = self.labels[signal_idx]
        return points, sdf, label, signal_idx

class ShapeNetSDFH5Chunked(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        seed: int = 42,
        debug: bool = False,
        train: bool = True,
        chunk_size=5000,
    ):
        self.root = root
        self.debug = debug
        if train:
            self.file_path = os.path.join(self.root, "shapenet_train.h5")
        else:
            self.file_path = os.path.join(self.root, "shapenet_test.h5")

        self.data = h5py.File(self.file_path, "r")
        self.num_signals = self.data["indices"].shape[0]
        self.points_per_signal = self.data["points"].shape[0] // self.num_signals
        self.total_num_points = self.data["points"].shape[0]
        if self.debug:
            if train:
                self.num_signals = 2560
            else:
                self.num_signals = 64
            self.points = self.data["points"][:self.num_signals*self.points_per_signal]
            self.sdf = self.data["sdf"][:self.num_signals*self.points_per_signal]
            self.total_num_points = self.num_signals*self.points_per_signal
        else:
            self.points = self.data["points"][:]
            self.sdf = self.data["sdf"][:]
            
        self.chunk_size = chunk_size
        self.num_chunks_per_signal = self.points_per_signal // self.chunk_size
        self.points = self.points.astype(np.float32).reshape(
            -1, self.chunk_size, 3
        )
        self.sdf = self.sdf.astype(np.float32).reshape(
            -1, self.chunk_size
        )
        self.total_num_chunks = self.points.shape[0]
        
        self.labels = self.data["labels"][:]

    def __len__(self):
        return self.total_num_chunks

    def __getitem__(self, idx):
        signal_idx, chunk_idx = divmod(idx, self.num_chunks_per_signal)
        points = self.points[idx]
        sdf = self.sdf[idx]

        # # Get label
        label = self.labels[signal_idx]
        return points, sdf, label, signal_idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_path = "/scratch-shared/papas/"
    dataset = ShapeNetSDFH5(data_path, train=True, debug=True)
    
    print(dataset[0])
    
    # plot the isocountours
    contour_level = 0.0
    points, sdf, label = dataset[10]
    contour_points_mask = np.abs(sdf - contour_level) < 0.005
    plt.figure()
    # create a 3D plot
    ax = plt.axes(projection='3d')
    ax.set_title(label)
    ax.scatter(points[contour_points_mask, 0], points[contour_points_mask, 2], points[contour_points_mask, 1], c=sdf[contour_points_mask], cmap='magma', s=0.5)
    # rotate the plot
    ax.view_init(30, 30)
    # make sure the aspect ratio is correct
    ax.set_box_aspect([2,1,1])
    plt.savefig("contour.png")
    