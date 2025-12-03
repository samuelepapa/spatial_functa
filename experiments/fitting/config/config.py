from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    # Experiment directory
    config.experiments_root = '/ivi/xfs/spapa/spatial_functa/'
    config.experiment_name = 'cifar10_functa'
    config.seed = 0

    config.train = ConfigDict()
    config.train.batch_size = 128
    config.train.num_steps = 50_000
    config.train.clip_grads = None
    config.train.num_minibatches = 1

    # Meta-learning
    config.train.inner_learning_rate = 1e-2  # only used if learn_lrs is False
    config.train.inner_lr_init_range = (0.005, 0.1)
    config.train.inner_lr_clip_range = (0.0, 1.0)
    config.train.inner_steps = 3
    config.train.inner_lr_multiplier = 32.0


    # Logging
    config.train.log_steps = ConfigDict()
    config.train.log_steps.loss = 200
    config.train.log_steps.image = 2500

    # Profiling
    # Set to None and comment the following two lines to disable
    config.train.profiler = ConfigDict()  
    config.train.profiler.start_step = 200
    config.train.profiler.end_step = 300

    # Checkpointing
    config.train.checkpointing = ConfigDict()
    config.train.checkpointing.checkpoint_dir_name = "ckpts"
    config.train.checkpointing.checkpoint_interval = 2500 # in number of steps
    config.train.checkpointing.tracked_metric = "psnr"

    # Validation
    config.valid = ConfigDict()
    config.valid.val_interval = 5000 # in number of steps

    return config
