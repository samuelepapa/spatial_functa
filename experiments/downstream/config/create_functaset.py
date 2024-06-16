
import time

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.seed = 0

    # Experiment directory to store artifacts
    config.experiment_dir = f'outputs/downstream_functa_{time.strftime("%Y%m%d-%H%M%S")}/'
    # Root folder of the checkpoint, the checkpoint should be in 
    # config.functa_model_dir / ckpts / best_psnr
    config.functa_model_dir = 'outputs/functa_20240615-195549/'

    config.functaset = ConfigDict()
    config.functaset.path = "spatial_cifar10" # this is relative to the DATA_PATH env var    
    config.functaset.functa_bank_size_per_batch = 10000 # number of batches per h5py file created

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
    # Set to None and comment the following two lines to disable
    config.train.profiler = ConfigDict()  
    config.train.profiler.start_step = 200
    config.train.profiler.end_step = 300

    return config
