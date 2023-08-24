from typing import Optional

from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
import cv2
import numpy as np
import torch

from armory.art_experimental.attacks.carla_obj_det_utils import (
    PatchMask,
    fetch_image_from_file_or_url,
    linear_depth_to_rgb,
    linear_to_log,
    log_to_linear,
    rgb_depth_to_linear,
)
from armory.logs import log


class CARLAAdversarialPatchPyTorch(AdversarialPatchPyTorch):
    """
    Apply patch attack to RGB channels and (optionally) masked PGD attack to depth channels.
    """

    def __init__(self, estimator, **kwargs):
        # Maximum depth perturbation from a flat patch
        self.depth_delta_meters = kwargs.pop("depth_delta_meters", 3)
        self.learning_rate_depth = kwargs.pop("learning_rate_depth", 0.0001)
        self.depth_perturbation = None
        self.min_depth = None
        self.max_depth = None
        self.patch_base_image = kwargs.pop("patch_base_image", None)
        self.patch_mask = PatchMask.from_kwargs(kwargs.pop("patch_mask", None))

        # HSV bounds are user-defined to limit perturbation regions
        self.hsv_lower_bound = np.array(
            kwargs.pop("hsv_lower_bound", [0, 0, 0])
        )  # [0, 0, 0] means unbounded below
        self.hsv_upper_bound = np.array(
            kwargs.pop("hsv_upper_bound", [255, 255, 255])
        )  # [255, 255, 255] means unbounded above

        super().__init__(estimator=estimator, **kwargs)

    def create_initial_image(self, size, hsv_lower_bound, hsv_upper_bound):
        """
        Create initial patch based on a user-defined image and
        create perturbation mask based on HSV bounds
        """
        if not isinstance(self.patch_base_image, str):
            raise ValueError(
                "patch_base_image must be a string path to an image or a url to an image"
            )
        im = fetch_image_from_file_or_url(self.patch_base_image)
        im = cv2.resize(im, size)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # find the colors within the boundaries
        color_mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
        color_mask = np.expand_dims(color_mask, 2)
        # cv2.imwrite(
        #     "color_mask.png", color_mask
        # )  # visualize perturbable regions. Comment out if not needed.

        patch_base = np.transpose(im, (2, 0, 1))
        patch_base = patch_base / 255.0
        color_mask = np.transpose(color_mask, (2, 0, 1))
        color_mask = color_mask / 255.0
        return patch_base, color_mask

    def _train_step(
        self,
        images: "torch.Tensor",
        target: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        self.estimator.model.zero_grad()
        # only zero gradients when there is a non-pgd optimizer; pgd optimizer appears to perform better when gradients accumulate
        if self._optimizer_string == "Adam":
            self._optimizer_rgb.zero_grad(set_to_none=True)
            if images.shape[-1] == 6:
                self._optimizer_depth.zero_grad(set_to_none=True)
        loss = self._loss(images, target, mask)
        loss.backward(retain_graph=False)

        if self._optimizer_string == "pgd":
            patch_grads = self._patch.grad
            patch_gradients = (
                patch_grads.sign() * self.learning_rate * self.patch_color_mask
            )

            if images.shape[-1] == 6:
                depth_grads = self.depth_perturbation.grad
                if self.depth_type == "log":
                    depth_log = (
                        self.depth_perturbation
                        + depth_grads.sign() * self.learning_rate_depth
                    )
                else:
                    grads_linear = rgb_depth_to_linear(
                        depth_grads[:, 0, :, :],
                        depth_grads[:, 1, :, :],
                        depth_grads[:, 2, :, :],
                    )
                    depth_linear = rgb_depth_to_linear(
                        self.depth_perturbation[:, 0, :, :],
                        self.depth_perturbation[:, 1, :, :],
                        self.depth_perturbation[:, 2, :, :],
                    )
                    depth_linear = (
                        depth_linear + grads_linear.sign() * self.learning_rate_depth
                    )

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch + patch_gradients,
                    min=self.estimator.clip_values[0],
                    max=self.estimator.clip_values[1],
                )

                if images.shape[-1] == 6:
                    images_depth = torch.permute(images[:, :, :, 3:], (0, 3, 1, 2))
                    if self.depth_type == "log":
                        perturbed_images = torch.clamp(
                            images_depth + depth_log,
                            min=self.min_depth,
                            max=self.max_depth,
                        )
                    else:
                        images_depth_linear = rgb_depth_to_linear(
                            images_depth[:, 0, :, :],
                            images_depth[:, 1, :, :],
                            images_depth[:, 2, :, :],
                        )
                        depth_linear = torch.clamp(
                            images_depth_linear + depth_linear,
                            min=self.min_depth,
                            max=self.max_depth,
                        )
                        depth_r, depth_g, depth_b = linear_depth_to_rgb(depth_linear)
                        perturbed_images = torch.stack(
                            [depth_r, depth_g, depth_b], dim=1
                        )
                    self.depth_perturbation[:] = perturbed_images - images_depth

        else:
            self._optimizer_rgb.step()
            if images.shape[-1] == 6:
                self._optimizer_depth.step()

            with torch.no_grad():
                self._patch[:] = torch.clamp(
                    self._patch,
                    min=self.estimator.clip_values[0],
                    max=self.estimator.clip_values[1],
                )

                if images.shape[-1] == 6:
                    images_depth = torch.permute(images[:, :, :, 3:], (0, 3, 1, 2))
                    if self.depth_type == "log":
                        perturbed_images = torch.clamp(
                            images_depth + self.depth_perturbation,
                            min=self.min_depth,
                            max=self.max_depth,
                        )
                    else:
                        images_depth_linear = rgb_depth_to_linear(
                            images_depth[:, 0, :, :],
                            images_depth[:, 1, :, :],
                            images_depth[:, 2, :, :],
                        )
                        depth_linear = rgb_depth_to_linear(
                            self.depth_perturbation[:, 0, :, :],
                            self.depth_perturbation[:, 1, :, :],
                            self.depth_perturbation[:, 2, :, :],
                        )
                        depth_linear = torch.clamp(
                            images_depth_linear + depth_linear,
                            min=self.min_depth,
                            max=self.max_depth,
                        )
                        depth_r, depth_g, depth_b = linear_depth_to_rgb(depth_linear)
                        perturbed_images = torch.stack(
                            [depth_r, depth_g, depth_b], dim=1
                        )
                    self.depth_perturbation[:] = perturbed_images - images_depth

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

        images_rgb = images[:, :3, :, :]
        if images.shape[1] == 6:
            images_depth = images[:, 3:, :, :]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        self.image_shape = images_rgb.shape[1:]

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
                [width - pad_w_after - 1, pad_h_before],
                [width - pad_w_after - 1, height - pad_h_after - 1],
                [pad_w_before, height - pad_h_after - 1],
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
        foreground_mask = ~(
            ~foreground_mask * image_mask.bool()
        )  # ensure area perturbed in depth is consistent with area perturbed in RGB

        # Adjust green screen brightness
        v_avg = (
            0.5647  # average V value (in HSV) for the green screen, which is #00903a
        )
        green_screen = images_rgb * image_mask
        values, _ = torch.max(green_screen, dim=1, keepdim=True)
        values_ratio = values / v_avg
        values_ratio = torch.repeat_interleave(values_ratio, dim=1, repeats=3)

        patched_images = (
            images_rgb * inverted_mask
            + padded_patch * values_ratio * image_mask
            - padded_patch * values_ratio * foreground_mask * image_mask
            + images_rgb * foreground_mask * image_mask
        )

        patched_images = torch.clamp(
            patched_images,
            min=self.estimator.clip_values[0],
            max=self.estimator.clip_values[1],
        )

        if not self.estimator.channels_first:
            patched_images = torch.permute(patched_images, (0, 2, 3, 1))

        # Apply perturbation to depth channels
        if images.shape[1] == 6:
            perturbed_images = images_depth + self.depth_perturbation * ~foreground_mask

            perturbed_images = torch.clamp(
                perturbed_images,
                min=self.estimator.clip_values[0],
                max=self.estimator.clip_values[1],
            )

            if not self.estimator.channels_first:
                perturbed_images = torch.permute(perturbed_images, (0, 2, 3, 1))

            return torch.cat([patched_images, perturbed_images], dim=-1)

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
                3,
                patch_height,
                patch_width,
            )

            # Use this mask to embed patch into the background in the event of occlusion
            self.binarized_patch_mask = y_patch_metadata[i]["mask"]

            # Add patch mask to the image mask
            if self.patch_mask is not None:
                orig_patch_mask = self.binarized_patch_mask.copy()
                projected_mask = self.patch_mask.project(
                    self.binarized_patch_mask.shape, gs_coords, as_bool=True
                )
                # binarized_patch_mask already handled in loss function
                self.binarized_patch_mask *= projected_mask

            # Eval7 contains a mixture of patch locations.
            # Patches that lie flat on the sidewalk or street are constrained to 0.03m depth perturbation, and they are best used to create disappearance errors.
            # Patches located elsewhere (i.e., that do not impede pedestrian/vehicle motion) are constrained to 3m depth perturbation, and they are best used to create hallucinations.
            # Therefore, the depth perturbation bound for each patch is input-dependent.
            if x.shape[-1] == 6:
                if "max_depth_perturb_meters" in y_patch_metadata[i].keys():
                    self.depth_delta_meters = y_patch_metadata[i][
                        "max_depth_perturb_meters"
                    ]
                    log.info(
                        'This dataset contains input-dependent depth perturbation bounds, and the user-defined "depth_delta_meters" has been reset to {} meters'.format(
                            y_patch_metadata[i]["max_depth_perturb_meters"]
                        )
                    )

            # self._patch needs to be re-initialized with the correct shape
            if self.patch_base_image is not None:
                patch_init, patch_color_mask = self.create_initial_image(
                    (patch_width, patch_height),
                    self.hsv_lower_bound,
                    self.hsv_upper_bound,
                )
            else:
                patch_init = np.random.randint(0, 255, size=self.patch_shape) / 255
                patch_color_mask = np.ones_like(patch_init)

            self._patch = torch.tensor(
                patch_init, requires_grad=True, device=self.estimator.device
            )
            self.patch_color_mask = torch.Tensor(patch_color_mask).to(
                self.estimator.device
            )

            # initialize depth variables
            if x.shape[-1] == 6:
                # check if depth image is log-depth
                if np.all(x[i, :, :, 3] == x[i, :, :, 4]) and np.all(
                    x[i, :, :, 3] == x[i, :, :, 5]
                ):
                    self.depth_type = "log"
                    depth_linear = log_to_linear(x[i, :, :, 3:])
                    max_depth = linear_to_log(depth_linear + self.depth_delta_meters)
                    min_depth = linear_to_log(depth_linear - self.depth_delta_meters)
                    max_depth = np.transpose(np.minimum(1.0, max_depth), (2, 0, 1))
                    min_depth = np.transpose(np.maximum(0.0, min_depth), (2, 0, 1))
                else:
                    self.depth_type = "linear"
                    depth_linear = rgb_depth_to_linear(
                        x[i, :, :, 3], x[i, :, :, 4], x[i, :, :, 5]
                    )
                    max_depth = depth_linear + self.depth_delta_meters
                    min_depth = depth_linear - self.depth_delta_meters
                    max_depth = np.minimum(1000.0, max_depth)
                    min_depth = np.maximum(0.0, min_depth)

                self.max_depth = torch.tensor(
                    np.expand_dims(max_depth, axis=0),
                    dtype=torch.float32,
                    device=self.estimator.device,
                )
                self.min_depth = torch.tensor(
                    np.expand_dims(min_depth, axis=0),
                    dtype=torch.float32,
                    device=self.estimator.device,
                )
                self.depth_perturbation = torch.zeros(
                    1,
                    3,
                    x.shape[1],
                    x.shape[2],
                    requires_grad=True,
                    device=self.estimator.device,
                )

            if self._optimizer_string == "Adam":
                self._optimizer_rgb = torch.optim.Adam(
                    [self._patch], lr=self.learning_rate
                )
                if x.shape[-1] == 6:
                    self._optimizer_depth = torch.optim.Adam(
                        [self.depth_perturbation], lr=self.learning_rate_depth
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

            # Embed patch mask fill into masked region
            if self.patch_mask is not None:
                patched_image = self.patch_mask.fill_masked_region(
                    patched_image=patched_image,
                    projected_mask=projected_mask,
                    gs_coords=gs_coords,
                    patch_init=patch_init,
                    orig_patch_mask=orig_patch_mask,
                )

            patched_image = np.clip(
                patched_image,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            attacked_images.append(patched_image)

        return np.array(attacked_images)
