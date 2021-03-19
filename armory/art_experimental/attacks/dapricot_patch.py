from art.attacks.evasion import RobustDPatch, ProjectedGradientDescent
from armory.art_experimental.utils.dapricot_patch_utils import insert_patch
import numpy as np


class DApricotPatch(RobustDPatch):
    """
    Apply Masked PGD to image and video inputs,
    where images are assumed to have shape (NHWC)
    and video are assumed to have shape (NFHWC)
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y_object=None, y_patch_metadata=None, **generate_kwargs):

        if "threat_model" not in generate_kwargs:
            raise ValueError(
                "'threat_model' kwarg must be defined in attack config's"
                "'generate_kwargs' as one of ('physical', 'digital')"
            )
        elif generate_kwargs["threat_model"].lower() not in ("physical", "digital"):
            raise ValueError(
                f"'threat_model must be set to one of ('physical', 'digital'), not {generate_kwargs['threat_model']}."
            )

        else:
            threat_model = generate_kwargs["threat_model"].lower()

        num_imgs = x.shape[0]
        attacked_images = []

        from PIL import (
            Image,
        )  # temporary so we can view outputs, see bottom of loop below

        if threat_model == "digital":
            for i in range(num_imgs):
                gs_coords = y_patch_metadata[i]["gs_coords"]
                # shape = y_patch_metadata[0]["shape"]
                shape = "diamond"  # temporary, having trouble parsing "shape" key
                cc_gt = y_patch_metadata[i]["cc_ground_truth"]
                cc_scene = y_patch_metadata[i]["cc_scene"]

                patch = super().generate(np.expand_dims(x[i], axis=0))

                img_with_patch = insert_patch(
                    gs_coords,
                    x[i],
                    patch,
                    shape,
                    cc_gt,
                    cc_scene,
                    apply_realistic_effects=True,
                )
                attacked_images.append(img_with_patch)

                img_with_patch_pil = Image.fromarray(np.uint8(img_with_patch * 255.0))
                img_with_patch_pil.save(f"image_with_patch_{i}.jpg")

        else:
            pass  # TODO: threat_model is physical. Generate patch universal across the 3 cameras

        return np.array(attacked_images)
