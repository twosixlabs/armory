from art.attacks.evasion import ProjectedGradientDescent
import numpy as np


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

        if "ymin" in generate_kwargs:
            ymin = generate_kwargs["ymin"]
        else:
            raise ValueError("generate_kwargs did not define 'ymin'")

        if "xmin" in generate_kwargs:
            xmin = generate_kwargs["xmin"]
        else:
            raise ValueError("generate_kwargs did not define 'xmin'")

        assert x.ndim in [
            4,
            5,
        ], "This attack is designed for images (4-dim) and videos (5-dim)"

        channels_mask = generate_kwargs.get(
            "mask", np.ones(x.shape[-1], dtype=np.float32)
        )
        channels = np.where(channels_mask)[0]

        mask = np.zeros(shape=x.shape[1:], dtype=np.float32)
        if "patch_ratio" in generate_kwargs:
            patch_ratio = generate_kwargs["patch_ratio"]
            ymax = ymin + int(x.shape[-3] * patch_ratio ** 0.5)
            xmax = xmin + int(x.shape[-2] * patch_ratio ** 0.5)
            if video_input:
                mask[:, ymin:ymax, xmin:xmax, channels] = 1.0
            else:
                mask[ymin:ymax, xmin:xmax, channels] = 1.0
        elif "patch_height" in generate_kwargs and "patch_width" in generate_kwargs:
            patch_height = generate_kwargs["patch_height"]
            patch_width = generate_kwargs["patch_width"]
            if video_input:
                mask[
                    :, ymin : ymin + patch_height, xmin : xmin + patch_width, channels
                ] = 1.0
            else:
                mask[
                    ymin : ymin + patch_height, xmin : xmin + patch_width, channels
                ] = 1.0
        else:
            raise ValueError(
                "generate_kwargs did not define 'patch_ratio', or it did not define 'patch_height' and 'patch_width'"
            )
        return super().generate(x, y=y, mask=mask)
