from typing import Callable

import jax
import optax


def extract_learning_rate(
    learning_rate: Callable, opt_state: optax.TraceState, prev_states=None
):
    # accumulate all states that have a schedule by state, and extract the learning rate from the last
    if prev_states is None:
        prev_states = []

    for s in opt_state:
        if isinstance(s, optax.ScaleByScheduleState):
            prev_states.append(s)
        elif isinstance(s, tuple):
            extract_learning_rate(learning_rate, s, prev_states)

    if len(prev_states) > 0:
        return learning_rate(prev_states[-1].count)
    else:
        return None


def set_profiler(profiler_config, step, log_dir):
    # Profiling.
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=str(log_dir))
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()
