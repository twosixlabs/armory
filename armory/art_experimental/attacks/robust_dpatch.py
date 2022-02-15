from art.attacks.evasion import RobustDPatch
import numpy as np

from armory.logs import log


class RobustDPatch(RobustDPatch):
    """
    Generate and apply patch
    """

    def __init__(self, estimator, **kwargs):
        # allows for random patch location
        if "patch_location" not in kwargs:
            self.random_location = True
        else:
            self.random_location = False
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y=None, **generate_kwargs):
        if self.random_location:
            log.info("Selecting random coordinates for patch_location.")
            self.patch_location = (
                np.random.randint(int(x.shape[-3] - self.patch_shape[0])),
                np.random.randint(int(x.shape[-2] - self.patch_shape[1])),
            )
        super().generate(x, y=y, **generate_kwargs)
        return super().apply_patch(x)
