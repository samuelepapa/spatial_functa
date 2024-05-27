import time

from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Experiment directory
    config.experiment_dir = f'outputs/functa_{time.strftime("%Y%m%d-%H%M%S")}/'

    config.train = ConfigDict()
    config.train.batch_size = 64
    config.train.inner_learning_rate = 1e-2  # only used if learn_lrs is False
    config.train.inner_lr_init_range = (0.005, 0.1)
    config.train.inner_lr_clip_range = (0.0, 1.0)
    config.train.inner_steps = 3
    config.train.num_steps = 5e5
    config.train.log_steps = ConfigDict()
    config.train.log_steps.loss = 200
    config.train.log_steps.image = 2500
    config.train.profiler = ConfigDict()  # None
    config.train.profiler.start_step = 200
    config.train.profiler.end_step = 300
    config.train.clip_grads = None
    config.train.num_minibatches = 1
    config.train.log_dir = config.experiment_dir
    config.train.checkpointing = ConfigDict()
    config.train.checkpointing.checkpoint_dir = config.experiment_dir + "ckpts/"
    config.train.checkpointing.checkpoint_interval = 2500
    config.train.checkpointing.tracked_metric = "psnr"

    config.valid = ConfigDict()
    config.valid.val_interval = 1000

    config.seed = 0

    # Other configuration options...
    # config.learning_rate = 0.001
    # config.batch_size = 32

    return config
