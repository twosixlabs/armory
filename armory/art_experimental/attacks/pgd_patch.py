from art.attacks.evasion import ProjectedGradientDescent
import numpy as np


class PGDPatch(ProjectedGradientDescent):
    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y=None, **generate_kwargs):
        if "ymin" in generate_kwargs:
            ymin = generate_kwargs["ymin"]
        else:
            raise ValueError("generate_kwargs did not define 'ymin'")

        if "xmin" in generate_kwargs:
            xmin = generate_kwargs["xmin"]
        else:
            raise ValueError("generate_kwargs did not define 'xmin'")

        mask = np.zeros(shape=x.shape[1:])
        if "patch_ratio" in generate_kwargs:
            patch_ratio = generate_kwargs["patch_ratio"]
            ymax = ymin + int(x.shape[1] * patch_ratio ** 0.5)
            xmax = xmin + int(x.shape[2] * patch_ratio ** 0.5)
            mask[ymin:ymax, xmin:xmax, :] = 1
        elif "patch_height" in generate_kwargs and "patch_width" in generate_kwargs:
            patch_height = generate_kwargs["patch_height"]
            patch_width = generate_kwargs["patch_width"]
            mask[ymin : ymin + patch_height, xmin : xmin + patch_width, :] = 1
        else:
            raise ValueError(
                "generate_kwargs did not define 'patch_ratio', or it did not define 'patch_height' and 'patch_width'"
            )
        return super().generate(x, y=y, mask=mask)
