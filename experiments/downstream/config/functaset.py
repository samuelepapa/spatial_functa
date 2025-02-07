from ml_collections import ConfigDict

FUNCTASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
    },
}

def get_config(name: str = "cifar10") -> ConfigDict:

    config = ConfigDict()

    config.name = name
    config.path = "spatial_cifar10" # this is relative to the DATA_PATH env var    
    
    config.num_classes = FUNCTASET_CONFIGS[name]["num_classes"]
    config.num_workers = 8
    
    return config