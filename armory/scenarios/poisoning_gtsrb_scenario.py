"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import numpy as np
from PIL import ImageOps, Image

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


def gtsrb_scenario_preprocessing(batch):
    img_size = 48
    img_out = []
    quantization = 255.0
    for im in batch:
        img_eq = ImageOps.equalize(Image.fromarray(im))
        width, height = img_eq.size
        min_side = min(img_eq.size)
        center = width // 2, height // 2

        left = center[0] - min_side // 2
        top = center[1] - min_side // 2
        right = center[0] + min_side // 2
        bottom = center[1] + min_side // 2

        img_eq = img_eq.crop((left, top, right, bottom))
        img_eq = np.array(img_eq.resize([img_size, img_size])) / quantization

        img_out.append(img_eq)

    return np.array(img_out, dtype=np.float32)


class GTSRB(Poison):
    def set_dataset_kwargs(self):
        super().set_dataset_kwargs()
        self.dataset_kwargs["preprocessing_fn"] = gtsrb_scenario_preprocessing

    def set_random_seed_tensorflow(self):
        # TODO: Handle automatically
        if not self.config["sysconfig"].get("use_gpu"):
            conf = ConfigProto(intra_op_parallelism_threads=1)
            set_session(Session(config=conf))
        set_random_seed(self.seed)

    def set_random_seed(self):
        super().set_random_seed()
        self.set_random_seed_tensorflow()
