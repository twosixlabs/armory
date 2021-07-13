from art.attacks.evasion import ProjectedGradientDescent
import numpy as np


class FrameBorderPatch(ProjectedGradientDescent):
    """
    Apply Masked PGD to video inputs, where only the
    video frame is allowed to be perturbed.
    Each video is assumed to have shape (NFHWC).
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y=None, patch_ratio=None, **kwargs):
        if patch_ratio is None:
            raise ValueError("generate_kwargs did not define 'patch_ratio'")
        if x.ndim != 5:
            raise ValueError("This attack is designed for videos (5-dim)")
        width = x.shape[3]
        height = x.shape[2]

        t1 = (
            2 * (width + height)
            + (4 * (width + height) ** 2 - 16 * (patch_ratio * width * height)) ** 0.5
        ) / 8
        t2 = (
            2 * (width + height)
            - (4 * (width + height) ** 2 - 16 * (patch_ratio * width * height)) ** 0.5
        ) / 8
        thickness = int(min(t1, t2))

        if (width - 2 * thickness) * (height - 2 * thickness) < (
            1 - patch_ratio
        ) * width * height:
            raise ValueError("patch_ratio does not match height and width")

        mask = np.ones(shape=x.shape[1:], dtype=np.float32)

        mask[:, thickness : height - thickness, thickness : width - thickness, :] = 0.0

        return super().generate(x, y=y, mask=mask, **kwargs)
