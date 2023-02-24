import os
from typing import Optional

from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
import cv2
import numpy as np
import torch

from armory.art_experimental.attacks.carla_obj_det_utils import (
    get_avg_depth_value,
    linear_depth_to_rgb,
    linear_to_log,
    log_to_linear,
)
from armory.logs import log


class CARLAAdversarialPatchPyTorch(AdversarialPatchPyTorch):
    def __init__(self, estimator, **kwargs):

        # Maximum depth perturbation from a flat patch
        self.depth_delta_meters = kwargs.pop("depth_delta_meters", 0.03)
        self.learning_rate_depth = kwargs.pop("learning_rate_depth", 0.0001)
        self.max_depth_r = None
        self.min_depth_r = None
        self.max_depth_g = None
        self.min_depth_g = None
        self.max_depth_b = None
        self.min_depth_b = None

        self.patch_base_image = kwargs.pop("patch_base_image", None)

        super().__init__(estimator=estimator, **kwargs)

    def create_initial_image(self, size):
        """
        Create initial patch based on a user-defined image
        """
        module_path = globals()["__file__"]
        # user-defined image is assumed to reside in the same location as the attack module
        patch_base_image_path = os.path.abspath(
            os.path.join(os.path.join(module_path, "../"), self.patch_base_image)
        )

        im = cv2.imread(patch_base_image_path)
        im = cv2.resize(im, size)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        patch_base = np.transpose(im, (2, 0, 1))
        patch_base = patch_base / 255.0
        return patch_base

    def _train_step(
        self,
        images: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        self.estimator.model.zero_grad()
        loss = self._loss(images, target, mask)
        loss.backward(retain_graph=True)

        if self._optimizer_string == "pgd":
            patch_grads = self._patch.grad
            if patch_grads.shape[0] == 6:
                patch_grads[3:, :, :] = torch.mean(
                    patch_grads[3:, :, :], dim=0, keepdim=True
                )

            gradients = patch_grads.sign() * self.learning_rate
            if patch_grads.shape[0] == 6:
                gradients[3:, :, :] = (
                    gradients[3:, :, :] / self.learning_rate * self.learning_rate_depth
                )

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch + gradients,
                    min=self.estimator.clip_values[0],
                    max=self.estimator.clip_values[1],
                )

                if self._patch.shape[0] == 6:
                    if self.depth_type == "linear":
                        self._patch[3, :, :] = torch.clamp(
                            self._patch[3, :, :],
                            min=self.min_depth_r,
                            max=self.max_depth_r,
                        )
                        self._patch[4, :, :] = torch.clamp(
                            self._patch[4, :, :],
                            min=self.min_depth_g,
                            max=self.max_depth_g,
                        )
                        self._patch[5, :, :] = torch.clamp(
                            self._patch[5, :, :],
                            min=self.min_depth_b,
                            max=self.max_depth_b,
                        )
                    elif self.depth_type == "log":
                        self._patch[3:, :, :] = torch.clamp(
                            self._patch[3:, :, :],
                            min=self.min_depth,
                            max=self.max_depth,
                        )
                    else:
                        raise ValueError(
                            f"Expected depth_type in ('log', 'linear'). Found {self.depth_type}"
                        )
        else:
            raise ValueError(
                "Adam optimizer for CARLA Adversarial Patch not supported."
            )

        return loss

    def _get_circular_patch_mask(
        self, nb_samples: int, sharpness: int = 40
    ) -> "torch.Tensor":
        """
        Return a circular patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        image_mask = np.ones(
            (self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])
        )

        image_mask = np.expand_dims(image_mask, axis=0)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask = torch.Tensor(np.array(image_mask)).to(self.estimator.device)
        image_mask = torch.stack([image_mask] * nb_samples, dim=0)
        return image_mask

    def _random_overlay(
        self,
        images: "torch.Tensor",
        patch: "torch.Tensor",
        scale: Optional[float] = None,
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]
        import torchvision

        # Ensure channels-first
        if not self.estimator.channels_first:
            images = torch.permute(images, (0, 3, 1, 2))

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images.shape[1:]

        smallest_image_edge_scale = min(
            self.image_shape[self.i_h] / image_mask.shape[self.i_h + 1],
            self.image_shape[self.i_w] / image_mask.shape[self.i_w + 1],
        )

        image_mask = torchvision.transforms.functional.resize(
            img=image_mask,
            size=(
                int(smallest_image_edge_scale * image_mask.shape[self.i_h + 1]),
                int(smallest_image_edge_scale * image_mask.shape[self.i_w + 1]),
            ),
            interpolation=2,
        )

        pad_h_before = int(
            (self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2
        )
        pad_h_after = int(
            self.image_shape[self.i_h]
            - pad_h_before
            - image_mask.shape[self.i_h_patch + 1]
        )

        pad_w_before = int(
            (self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2
        )
        pad_w_after = int(
            self.image_shape[self.i_w]
            - pad_w_before
            - image_mask.shape[self.i_w_patch + 1]
        )

        image_mask = torchvision.transforms.functional.pad(
            img=image_mask,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(
                image_mask, dim=1, repeats=self.input_shape[0]
            )

        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        padded_patch = torchvision.transforms.functional.resize(
            img=padded_patch,
            size=(
                int(smallest_image_edge_scale * padded_patch.shape[self.i_h + 1]),
                int(smallest_image_edge_scale * padded_patch.shape[self.i_w + 1]),
            ),
            interpolation=2,
        )

        padded_patch = torchvision.transforms.functional.pad(
            img=padded_patch,
            padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
            fill=0,
            padding_mode="constant",
        )

        if self.nb_dims == 4:
            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(
                padded_patch, dim=1, repeats=self.input_shape[0]
            )

        padded_patch = padded_patch.float()

        image_mask_list = []
        padded_patch_list = []

        for i_sample in range(nb_samples):

            image_mask_i = image_mask[i_sample]

            height = padded_patch.shape[self.i_h + 1]
            width = padded_patch.shape[self.i_w + 1]

            startpoints = [
                [pad_w_before, pad_h_before],
                [width - pad_w_after, pad_h_before],
                [width - pad_w_after, height - pad_h_after],
                [pad_w_before, height - pad_h_after],
            ]
            endpoints = self.gs_coords[
                i_sample
            ]  # [topleft, topright, botright, botleft]
            enlarged_coords = np.copy(
                endpoints
            )  # enlarge the green screen coordinates a bit to fully cover the screen
            pad_amt_x = int(0.03 * (enlarged_coords[2, 0] - enlarged_coords[0, 0]))
            pad_amt_y = int(0.03 * (enlarged_coords[2, 1] - enlarged_coords[0, 1]))
            enlarged_coords[0, 0] -= pad_amt_x
            enlarged_coords[0, 1] -= pad_amt_y
            enlarged_coords[1, 0] += pad_amt_x
            enlarged_coords[1, 1] -= pad_amt_y
            enlarged_coords[2, 0] += pad_amt_x
            enlarged_coords[2, 1] += pad_amt_y
            enlarged_coords[3, 0] -= pad_amt_x
            enlarged_coords[3, 1] += pad_amt_y
            endpoints = enlarged_coords

            image_mask_i = torchvision.transforms.functional.perspective(
                img=image_mask_i,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=2,
                fill=0,  # None
            )

            image_mask_list.append(image_mask_i)

            padded_patch_i = padded_patch[i_sample]

            padded_patch_i = torchvision.transforms.functional.perspective(
                img=padded_patch_i,
                startpoints=startpoints,
                endpoints=endpoints,
                interpolation=2,
                fill=0,  # None
            )

            padded_patch_list.append(padded_patch_i)

        image_mask = torch.stack(image_mask_list, dim=0)
        padded_patch = torch.stack(padded_patch_list, dim=0)
        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(
                self.estimator.device
            )
            - image_mask
        )

        foreground_mask = torch.all(
            torch.tensor(self.binarized_patch_mask == 0), dim=-1, keepdim=True
        ).to(self.estimator.device)
        foreground_mask = torch.permute(foreground_mask, (2, 0, 1))
        foreground_mask = torch.unsqueeze(foreground_mask, dim=0)

        # Adjust green screen brightness
        v_avg = (
            0.5647  # average V value (in HSV) for the green screen, which is #00903a
        )
        green_screen = images * image_mask
        values, _ = torch.max(green_screen[:, :3, :, :], dim=1, keepdim=True)
        values_ratio = values / v_avg
        values_ratio = torch.repeat_interleave(values_ratio, dim=1, repeats=3)
        if images.shape[1] == 6:
            values_ratio_depth = torch.ones(values_ratio.size()).to(
                self.estimator.device
            )  # no brightness adjustment for depth
            values_ratio = torch.cat((values_ratio, values_ratio_depth), dim=1)

        patched_images = (
            images * inverted_mask
            + padded_patch * values_ratio * image_mask
            - padded_patch * values_ratio * foreground_mask * image_mask
            + images * foreground_mask * image_mask
        )
        patched_images = torch.clamp(
            patched_images,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))

        return patched_images

    def generate(self, x, y=None, y_patch_metadata=None):
        """
        param x: Sample images. For single-modality, shape=(NHW3). For multimodality, shape=(NHW6)
        param y: [Optional] Sample labels. List of dictionaries,
            ith dictionary contains bounding boxes, class labels, and class scores
        param y_patch_metadata: Patch metadata. List of N dictionaries, ith dictionary contains patch metadata for x[i]
        """

        if x.shape[0] > 1:
            log.info("To perform per-example patch attack, batch size must be 1")
        assert x.shape[-1] in [3, 6], "x must have either 3 or 6 color channels"

        num_imgs = x.shape[0]
        attacked_images = []
        for i in range(num_imgs):
            # Adversarial patch attack, when used for object detection, requires ground truth
            y_gt = dict()
            y_gt["labels"] = y[i]["labels"]
            non_patch_idx = np.where(
                y_gt["labels"] != 4
            )  # exclude the patch class, which doesn't exist in the training data
            y_gt["boxes"] = y[i]["boxes"][non_patch_idx]
            y_gt["labels"] = y_gt["labels"][non_patch_idx]
            y_gt["scores"] = np.ones(len(y_gt["labels"]), dtype=np.float32)

            gs_coords = y_patch_metadata[i]["gs_coords"]  # patch coordinates
            self.gs_coords = [gs_coords]
            patch_width = np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0])
            patch_height = np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1])
            self.patch_shape = (
                x.shape[-1],
                patch_height,
                patch_width,
            )

            # Use this mask to embed patch into the background in the event of occlusion
            self.binarized_patch_mask = y_patch_metadata[i]["mask"]

            # self._patch needs to be re-initialized with the correct shape
            if self.patch_base_image is not None:
                self.patch_base = self.create_initial_image(
                    (patch_width, patch_height),
                )
                if x.shape[-1] == 3:
                    patch_init = self.patch_base
                else:
                    patch_init = np.vstack(
                        (
                            self.patch_base,
                            np.random.randint(0, 255, size=self.patch_base.shape) / 255,
                        )
                    )
            else:
                patch_init = np.random.randint(0, 255, size=self.patch_shape) / 255

            if (
                self.patch_shape[0] == 6
            ):  # initialize depth patch with average depth value of green screen.

                # check if depth image is log-depth
                if (
                    x.shape[-1] == 6
                    and np.all(x[i, :, :, 3] == x[i, :, :, 4])
                    and np.all(x[i, :, :, 3] == x[i, :, :, 5])
                ):
                    self.depth_type = "log"
                    if "avg_patch_depth" in y_patch_metadata[i]:  # for Eval 5+ metadata
                        avg_patch_depth = y_patch_metadata[i]["avg_patch_depth"]
                    else:
                        # backward compatible with Eval 4 metadata
                        avg_patch_depth = get_avg_depth_value(x[i][:, :, 3], gs_coords)

                    avg_patch_depth_meters = log_to_linear(avg_patch_depth)
                    max_depth_meters = avg_patch_depth_meters + self.depth_delta_meters
                    min_depth_meters = avg_patch_depth_meters - self.depth_delta_meters
                    self.max_depth = linear_to_log(max_depth_meters)
                    self.min_depth = linear_to_log(min_depth_meters)
                    patch_init[3:, :, :] = avg_patch_depth
                else:
                    self.depth_type = "linear"
                    if "avg_patch_depth" in y_patch_metadata[i]:
                        avg_patch_depth = y_patch_metadata[i]["avg_patch_depth"]
                    else:
                        raise ValueError(
                            "Dataset does not contain patch metadata for average patch depth"
                        )
                    max_depth = avg_patch_depth + self.depth_delta_meters
                    min_depth = avg_patch_depth - self.depth_delta_meters
                    (
                        self.max_depth_r,
                        self.max_depth_g,
                        self.max_depth_b,
                    ) = linear_depth_to_rgb(max_depth)
                    (
                        self.min_depth_r,
                        self.min_depth_g,
                        self.min_depth_b,
                    ) = linear_depth_to_rgb(min_depth)
                    rv, gv, bv = linear_depth_to_rgb(avg_patch_depth)
                    patch_init[3, :, :] = rv
                    patch_init[4, :, :] = gv
                    patch_init[5, :, :] = bv

            self._patch = torch.tensor(
                patch_init, requires_grad=True, device=self.estimator.device
            )

            patch, _ = super().generate(np.expand_dims(x[i], axis=0), y=[y_gt])

            # Patch image
            x_tensor = torch.tensor(np.expand_dims(x[i], axis=0)).to(
                self.estimator.device
            )
            patched_image = (
                self._random_overlay(
                    images=x_tensor, patch=self._patch, scale=None, mask=None
                )
                .detach()
                .cpu()
                .numpy()
            )
            patched_image = np.squeeze(patched_image, axis=0)

            # Embed patch into background
            patched_image[np.all(self.binarized_patch_mask == 0, axis=-1)] = x[i][
                np.all(self.binarized_patch_mask == 0, axis=-1)
            ]

            patched_image = np.clip(
                patched_image,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            attacked_images.append(patched_image)

        return np.array(attacked_images)
