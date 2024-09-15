import optax
from ml_collections.config_dict import ConfigDict


def build_lr_scheduler(scheduler_config: ConfigDict, num_steps: int = 0):
    """Build learning rate schedule from config.

    By default, it supports constant, cosine decay, exponential decay, and warmup cosine decay.
    To add custom learning rate schedules, overwrite the function build_extra_lr_scheduler.

    Args:
        num_steps (int, optional): Number of steps for the learning rate schedule.

    Returns:
        Callable: Learning rate schedule function.
    """
    # Build learning rate schedule
    lr = scheduler_config.lr
    scheduler_name = scheduler_config.get("name", None)
    decay_steps = scheduler_config.get("decay_steps", num_steps)
    lr_schedule = None
    if scheduler_name is None or scheduler_name == "constant":
        lr_schedule = optax.constant_schedule(lr)
    elif scheduler_name == "cosine_decay":
        assert decay_steps > 0, "decay_steps must be positive"
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=decay_steps,
            alpha=scheduler_config.get("alpha", 0.0),
        )
    elif scheduler_name == "exponential_decay":
        # Exponential decay with cooldown and warmup
        cooldown = scheduler_config.get("cooldown_steps", 0)
        warmup = scheduler_config.get("warmup_steps", 0)
        num_schedule_steps = decay_steps - cooldown - warmup
        if scheduler_config.get("decay_rate", None) is not None:
            decay_rate = scheduler_config.decay_rate
        else:
            assert decay_steps > 0, "decay_steps must be positive"
            if scheduler_config.get("end_lr", None) is not None:
                end_lr_factor = scheduler_config.end_lr / lr
            elif scheduler_config.get("end_lr_factor", None) is not None:
                end_lr_factor = scheduler_config.end_lr_factor
            else:
                raise ValueError("Either end_lr or end_lr_factor must be specified.")
            decay_rate = end_lr_factor ** (1.0 / num_schedule_steps)
        lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            decay_rate=decay_rate,
            end_value=scheduler_config.get("end_value", 0.0),
            warmup_steps=warmup,
            transition_steps=scheduler_config.get("transition_steps", 1),
            staircase=scheduler_config.get("staircase", False),
        )
        if cooldown > 0:
            assert decay_steps > 0, "decay_steps must be positive"
            end_lr = lr * (decay_rate**num_schedule_steps)
            lr_schedule = optax.join_schedules(
                schedules=[
                    lr_schedule,
                    optax.linear_schedule(
                        init_value=end_lr,
                        end_value=0.0,
                        transition_steps=cooldown,
                    ),
                ],
                boundaries=[decay_steps - cooldown],
            )
    elif scheduler_name == "warmup_cosine_decay":
        assert decay_steps > 0, "decay_steps must be positive"
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            decay_steps=decay_steps,
            warmup_steps=scheduler_config.warmup_steps,
            end_value=scheduler_config.get("end_value", 0.0),
        )
    elif scheduler_name == "warmup_constant":
        lr_schedule = optax.schedules.warmup_constant_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=scheduler_config.warmup_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler {scheduler_name}")

    return lr_schedule
