from ml_collections import config_dict

DATASET_CONFIGS = {
    "cifar10": {
        "resolution": 32,
        "num_channels": 3,
        "num_classes": 10,
        "coords_dim": 2,
        "data_type": "image",
    },
    "mnist": {
        "resolution": 28,
        "num_channels": 1,
        "num_classes": 10,
        "coords_dim": 2,
        "data_type": "image",
    },
    "shapenet": {
        "num_channels": 1,
        "num_classes": 55,
        "coords_dim": 3,
        "data_type": "sdf",
        "resolution": 50_000,
        "total_points": 150_000,
    },
    "shapenet_batched": {
        "num_channels": 1,
        "num_classes": 55,
        "coords_dim": 3,
        "data_type": "sdf",
        "resolution": 50_000,
        "total_points": 150_000,
    },
    "shapenet_chunked": {
        "num_channels": 1,
        "num_classes": 55,
        "coords_dim": 3,
        "data_type": "sdf",
        "resolution": 50_000,
        "total_points": 150_000,
    },
    "shapenet_10classes": {
        "num_channels": 1,
        "num_classes": 10,
        "coords_dim": 3,
        "data_type": "sdf",
        "resolution": 50_000,
        "total_points": 150_000,
    },
}


def get_config(name):
    config = config_dict.ConfigDict()

    # Dataset location
    config.name = name
    config.resolution = DATASET_CONFIGS[name]["resolution"]
    config.num_channels = DATASET_CONFIGS[name]["num_channels"]
    config.num_classes = DATASET_CONFIGS[name]["num_classes"]
    config.coords_dim = DATASET_CONFIGS[name]["coords_dim"]
    config.data_type = DATASET_CONFIGS[name]["data_type"]
    config.total_points = DATASET_CONFIGS[name].get("total_points", None)

    config.debug = False    
    config.prefetch = False
    config.sampling_mode = "full_image"
    config.num_augmentations = 1
    config.apply_agument = False
    config.num_workers = 0

    return config
