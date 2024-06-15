
import time

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Experiment directory
    config.experiment_dir = f'outputs/downstream_functa_{time.strftime("%Y%m%d-%H%M%S")}/'
    config.functa_model_dir = 'outputs/functa_20240611-112046/'
    config.functa_bank_size_per_batch = 1000
    config.functaset = ConfigDict()
    config.functaset.path = "spatial_cifar10" # this is relative to the DATA_PATH env var
    config.seed = 0

    config.train = ConfigDict()
    config.train.batch_size = 64
    config.train.clip_grads = None
    config.train.num_minibatches = 1

    # Logging
    config.train.log_dir = config.experiment_dir
    config.train.log_steps = ConfigDict()
    config.train.log_steps.loss = 200
    config.train.log_steps.image = 2500

    # Profiling
    # Set to None and comment the following two lines to disable
    config.train.profiler = ConfigDict()  
    config.train.profiler.start_step = 200
    config.train.profiler.end_step = 300

    return config
