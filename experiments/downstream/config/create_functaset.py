
import time
import os

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.seed = 0

    # Experiment directory to store artifacts
    config.experiment_dir = f'outputs/create_functaset_{time.strftime("%Y%m%d-%H%M%S")}_{os.uname().nodename}/'
    # Root folder of the checkpoint, the checkpoint should be in 
    # config.functa_model_dir / ckpts / best_psnr
    config.functa_model_dir = "/home/papas/project_folder/spatial_functa/spatial_functa_1NN_CIFAR10_4464_outerclip_01_innerclip01/"

    # Params used during fitting
    config.train = ConfigDict()
    config.train.batch_size = 256
    config.train.num_minibatches = 1

    # Logging
    config.train.log_dir = config.experiment_dir
    config.train.log_steps = ConfigDict()
    config.train.log_steps.loss = 200
    config.train.log_steps.image = 2500

    # Profiling
    # Set to None and comment the two lines that follow to disable
    config.train.profiler = None # ConfigDict()  
    # config.train.profiler.start_step = 200
    # config.train.profiler.end_step = 300

    return config

