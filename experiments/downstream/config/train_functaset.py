import time

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Experiment directory
    config.experiment_dir = f'outputs/downstream_functa_{time.strftime("%Y%m%d-%H%M%S")}/'
    config.seed = 0

    config.train = ConfigDict()
    config.train.batch_size = 512
    config.train.num_steps = 3e5
    config.train.clip_grads = None
    config.train.weight_decay = 1e-1
    config.train.num_minibatches = 1
    config.train.num_classes = 10

    config.train.label_smoothing = True
    config.train.label_smoothing_factor = 0.1
    config.train.normalizing_factor = 0.08  # divides the latents by this factor
    config.train.clip_grads = None
    config.train.exp_mov_avg_beta = 0.9999

    # Logging
    config.train.log_dir = config.experiment_dir
    config.train.log_steps = ConfigDict()
    config.train.log_steps.loss =200
    config.train.log_steps.image = 2500

    # Profiling
    # Set to None and comment the following two lines to disable
    config.train.profiler = ConfigDict()  
    config.train.profiler.start_step = 200
    config.train.profiler.end_step = 300

    # Checkpointing
    config.train.checkpointing = ConfigDict()
    config.train.checkpointing.checkpoint_dir = config.experiment_dir + "ckpts/"
    config.train.checkpointing.checkpoint_interval = 100000 # in number of steps
    config.train.checkpointing.tracked_metric = "acc"

    # Validation
    config.valid = ConfigDict()
    config.valid.val_interval = 10000 # in number of steps

    return config
