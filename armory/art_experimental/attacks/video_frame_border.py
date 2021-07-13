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

    def generate(self, x, y=None, **generate_kwargs):
        assert x.ndim == 5, "This attack is designed for videos (5-dim)"

        width = x.shape[3]
        height = x.shape[2]

        if "patch_ratio" in generate_kwargs:
            patch_ratio = generate_kwargs["patch_ratio"]
            t1 = (
                2 * (width + height)
                + (4 * (width + height) ** 2 - 16 * (patch_ratio * width * height))
                ** 0.5
            ) / 8
            t2 = (
                2 * (width + height)
                - (4 * (width + height) ** 2 - 16 * (patch_ratio * width * height))
                ** 0.5
            ) / 8
            thickness = int(min(t1, t2))

            assert (width - 2 * thickness) * (height - 2 * thickness) >= (
                1 - patch_ratio
            ) * width * height
        else:
            raise ValueError("generate_kwargs did not define 'patch_ratio'")

        mask = np.ones(shape=x.shape[1:], dtype=np.float32)

        mask[:, thickness : height - thickness, thickness : width - thickness, :] = 0.0

        return super().generate(x, y=y, mask=mask)
