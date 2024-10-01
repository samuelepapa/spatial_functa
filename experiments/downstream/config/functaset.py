from ml_collections import ConfigDict

FUNCTASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
    },
    "mnist": {
        "num_classes": 10,
    },
    "shapenet": {
        "num_classes": 10,
    }
}


def get_config(name: str = "mnist") -> ConfigDict:

    config = ConfigDict()

    config.name = name
    config.path = "spatial_mnist_256"  # this is relative to the DATA_PATH env var

    config.num_classes = FUNCTASET_CONFIGS[name]["num_classes"]
    config.num_workers = 8

    return config
