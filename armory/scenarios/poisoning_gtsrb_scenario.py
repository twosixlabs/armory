"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

try:
    from tensorflow import set_random_seed, ConfigProto, Session
    from tensorflow.keras.backend import set_session
except ImportError:
    from tensorflow.compat.v1 import (
        set_random_seed,
        ConfigProto,
        Session,
        disable_v2_behavior,
    )
    from tensorflow.compat.v1.keras.backend import set_session

    disable_v2_behavior()

from armory.scenarios.poison import Poison


class GTSRB(Poison):
    def set_random_seed_tensorflow(self):
        # TODO: Handle automatically
        if not self.config["sysconfig"].get("use_gpu"):
            conf = ConfigProto(intra_op_parallelism_threads=1)
            set_session(Session(config=conf))
        set_random_seed(self.seed)

    def set_random_seed(self):
        super().set_random_seed()
        self.set_random_seed_tensorflow()
