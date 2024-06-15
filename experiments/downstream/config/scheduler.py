from ml_collections import config_dict


def get_config(scheduler_name: str):
    config = config_dict.ConfigDict()

    # Model
    config.name = scheduler_name

    if scheduler_name == "constant":
        config.lr = 2e-6
    elif scheduler_name == "cosine_decay":
        config.lr = 3e-4
        config.decay_steps = 10000
        config.alpha = 0.0
    elif scheduler_name == "exponential_decay":
        config.lr = 1e-5
        config.decay_steps = 20000
        config.cooldown_steps = 0
        config.warmup_steps = 200
        config.end_lr = config_dict.FieldReference(1e-4)
        config.end_value = config.get_ref("end_lr")
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name}")

    return config
