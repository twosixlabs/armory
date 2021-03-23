import numpy as np
import random

try:
    from tensorflow import set_random_seed
except ImportError:
    from tensorflow.compat.v1 import (
        set_random_seed,
        disable_v2_behavior,
    )

    disable_v2_behavior()


def set_seeds(sys_config: dict, scenario_config: dict):
    seed = scenario_config.get("seed", 42)
    np.random.seed(seed)
    set_random_seed(seed)
    random.seed(seed)
