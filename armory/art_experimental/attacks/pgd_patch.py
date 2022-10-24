from art.attacks.evasion import ProjectedGradientDescent
import numpy as np

from armory.logs import log


class PGDPatch(ProjectedGradientDescent):
    """
    Apply Masked PGD to image and video inputs,
    where images are assumed to have shape (NHWC)
    and video are assumed to have shape (NFHWC)
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y=None, **generate_kwargs):
        video_input = generate_kwargs.get("video_input", False)

        if "patch_ratio" in generate_kwargs:
            patch_ratio = generate_kwargs["patch_ratio"]
            patch_height = int(x.shape[-3] * patch_ratio**0.5)
            patch_width = int(x.shape[-2] * patch_ratio**0.5)
        elif "patch_height" in generate_kwargs and "patch_width" in generate_kwargs:
            patch_height = generate_kwargs["patch_height"]
            patch_width = generate_kwargs["patch_width"]
        else:
            raise ValueError(
                "generate_kwargs did not define 'patch_ratio', or it did not define 'patch_height' and 'patch_width"
            )

        if "ymin" in generate_kwargs:
            ymin = generate_kwargs["ymin"]
        else:
            log.info("Selecting random value for patch ymin coordinate.")
            ymin = np.random.randint(int(x.shape[-3] - patch_height))

        if "xmin" in generate_kwargs:
            xmin = generate_kwargs["xmin"]
        else:
            log.info("Selecting random value for patch xmin coordinate.")
            xmin = np.random.randint(int(x.shape[-2] - patch_width))

        assert x.ndim in [
            4,
            5,
        ], "This attack is designed for images (4-dim) and videos (5-dim)"

        channels_mask = generate_kwargs.get(
            "mask", np.ones(x.shape[-1], dtype=np.float32)
        )
        channels = np.where(channels_mask)[0]

        mask = np.zeros(shape=x.shape[1:], dtype=np.float32)

        if video_input:
            mask[
                :, ymin : ymin + patch_height, xmin : xmin + patch_width, channels
            ] = 1.0
        else:
            mask[ymin : ymin + patch_height, xmin : xmin + patch_width, channels] = 1.0

        return super().generate(x, y=y, mask=mask)
