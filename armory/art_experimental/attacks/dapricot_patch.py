import numpy as np
import cv2

from art.attacks.evasion import RobustDPatch, ProjectedGradientDescent
from armory.art_experimental.utils.dapricot_patch_utils import (
    insert_patch,
    shape_coords,
    create_mask,
)


class DApricotPatch(RobustDPatch):
    """
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

        if threat_model == "digital":
            for i in range(num_imgs):
                gs_coords = y_patch_metadata[i]["gs_coords"]
                patch_x, patch_y = gs_coords[0]  # upper left coordinates of patch
                self.patch_location = (patch_x, patch_y)
                patch_width = np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0])
                patch_height = np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1])
                self.patch_shape = (patch_height, patch_width, 3)

                # self._patch needs to be re-initialized with the correct shape
                if self.estimator.clip_values is None:
                    self._patch = np.zeros(shape=self.patch_shape)
                else:
                    self._patch = (
                        np.random.randint(0, 255, size=self.patch_shape)
                        / 255
                        * (
                            self.estimator.clip_values[1]
                            - self.estimator.clip_values[0]
                        )
                        + self.estimator.clip_values[0]
                    )

                patch = super().generate(np.expand_dims(x[i], axis=0))

                patch_geometric_shape = (
                    y_patch_metadata[i]["shape"].tobytes().decode("utf-8")
                )
                img_with_patch = insert_patch(
                    gs_coords, x[i], patch, patch_geometric_shape,
                )
                attacked_images.append(img_with_patch)
        else:
            # generate patch using center image
            gs_coords_center_img = y_patch_metadata[1]["gs_coords"]
            patch_x, patch_y = gs_coords_center_img[
                0
            ]  # upper left coordinates of patch
            self.patch_location = (patch_x, patch_y)
            patch_width = np.max(gs_coords_center_img[:, 0]) - np.min(
                gs_coords_center_img[:, 0]
            )
            patch_height = np.max(gs_coords_center_img[:, 1]) - np.min(
                gs_coords_center_img[:, 1]
            )
            self.patch_shape = (patch_height, patch_width, 3)

            # self._patch needs to be re-initialized with the correct shape
            if self.estimator.clip_values is None:
                self._patch = np.zeros(shape=self.patch_shape)
            else:
                self._patch = (
                    np.random.randint(0, 255, size=self.patch_shape)
                    / 255
                    * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                    + self.estimator.clip_values[0]
                )

            patch = super().generate(np.expand_dims(x[1], axis=0))

            for i in range(num_imgs):
                gs_coords = y_patch_metadata[i]["gs_coords"]
                patch_geometric_shape = (
                    y_patch_metadata[i]["shape"].tobytes().decode("utf-8")
                )

                img_with_patch = insert_patch(
                    gs_coords, x[i], patch, patch_geometric_shape,
                )
                attacked_images.append(img_with_patch)
        return np.array(attacked_images)


class DApricotMaskedPGD(ProjectedGradientDescent):
    """
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

        if threat_model == "digital":
            for i in range(num_imgs):
                gs_coords = y_patch_metadata[i]["gs_coords"]
                shape = y_patch_metadata[i]["shape"].tobytes().decode("utf-8")
                img_mask = self._compute_image_mask(
                    x[i], y_object[i]["area"], gs_coords, shape
                )
                img_with_patch = super().generate(
                    np.expand_dims(x[i], axis=0), mask=img_mask
                )[0]
                attacked_images.append(img_with_patch)
        else:
            raise NotImplementedError(
                "physical threat model not available for masked PGD attack"
            )

        return np.array(attacked_images)

    def _compute_image_mask(self, x, gs_area, gs_coords, shape):
        gs_size = int(np.sqrt(gs_area))
        patch_coords = shape_coords(gs_size, gs_size, shape)
        h, status = cv2.findHomography(patch_coords, gs_coords)
        inscribed_patch_mask = create_mask(shape, gs_size, gs_size)
        img_mask = cv2.warpPerspective(
            inscribed_patch_mask, h, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC
        )
        return img_mask
