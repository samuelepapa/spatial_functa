
from ml_collections import ConfigDict

FUNCTASET_CONFIGS = {
    "spatial_cifar10": {
        "num_classes": 10,
    },
}

def get_config(name: str = "spatial_cifar10") -> ConfigDict:

    config = ConfigDict()

    config.name = name
    config.path = "spatial_cifar10" # this is relative to the DATA_PATH env var    
    
    config.num_classes = FUNCTASET_CONFIGS[name]["num_classes"]
    
    return config