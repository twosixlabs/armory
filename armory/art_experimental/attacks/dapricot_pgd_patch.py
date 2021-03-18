from art.attacks.evasion import ProjectedGradientDescent
import numpy as np


class DApricotPGDPatch(ProjectedGradientDescent):
    """
    Apply Masked PGD to image and video inputs,
    where images are assumed to have shape (NHWC)
    and video are assumed to have shape (NFHWC)
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y_object=None, y_patch_metadata=None, **generate_kwargs):



        if "threat_model" not in generate_kwargs:
            raise ValueError("'threat_model' kwarg must be defined in attack config's"
                             "'generate_kwargs' as one of ('physical', 'digital')")
        elif generate_kwargs["threat_model"].lower() not in ("physical", "digital"):
            raise ValueError(f"'threat_model must be set to one of ('physical', 'digital'), not {generate_kwargs['threat_model']}.")

        else:
            threat_model = generate_kwargs["threat_model"].lower()


        mask = np.zeros(shape=x.shape[1:], dtype=np.float32)


        attacked_images = []
        if threat_model == "digital":

            breakpoint()


        return super().generate(x, y=y, mask=mask)


