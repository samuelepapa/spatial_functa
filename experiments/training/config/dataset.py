from ml_collections import config_dict

DATASET_CONFIGS = {
    "cifar10": {
        "resolution": 32,
        "num_channels": 3,
        "num_classes": 10,
    },
    "mnist": {
        "resolution": 28,
        "num_channels": 1,
        "num_classes": 10,
    },
}


def get_config(name):
    config = config_dict.ConfigDict()

    # Dataset location
    config.name = name
    config.resolution = DATASET_CONFIGS[name]["resolution"]
    config.num_channels = DATASET_CONFIGS[name]["num_channels"]
    config.num_classes = DATASET_CONFIGS[name]["num_classes"]

    config.prefetch = False
    config.sampling_mode = "full_image"
    config.num_augmentations = 1
    config.apply_agument = False
    config.num_workers = 2

    return config
