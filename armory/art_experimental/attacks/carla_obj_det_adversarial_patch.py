import numpy as np
from armory.logs import log

from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
from typing import Optional, List, Union, Tuple, Dict
import torch
from armory.utils.evaluation import patch_method


class CARLAAdversarialPatchPyTorch(AdversarialPatchPyTorch):
    def __init__(self, estimator, **kwargs):

        # Maximum depth perturbation from a flat patch
        self.depth_delta_meters = kwargs.pop("depth_delta_meters", 3)
        self.learning_rate_depth = kwargs.pop("learning_rate_depth", 0.0001)
        self.max_depth = None
        self.min_depth = None

        super().__init__(estimator=estimator, **kwargs)

        # Overwrite the _get_loss() function because the default version cannot handle input tensor with 6 channels,
        # i.e., https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/3683fd6fbf97bc97c1861b8f9ff19bfdc026885c/art/estimators/object_detection/python_object_detector.py#L217
        @patch_method(estimator)
        def _get_losses(
            self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
        ) -> Tuple[
            Dict[str, "torch.Tensor"], List["torch.Tensor"], List["torch.Tensor"]
        ]:
            """
            Get the loss tensor output of the model including all preprocessing.

            :param x: Samples of shape (nb_samples, height, width, nb_channels).
            :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as
                    follows:

                    - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                                0 <= y1 < y2 <= H.
                    - labels (Int64Tensor[N]): the labels for each image
            :return: Loss gradients of the same shape as `x`.
            """
            import torch  # lgtm [py/repeated-import]
            import torchvision  # lgtm [py/repeated-import]

            self._model.train()

            # Apply preprocessing
            if self.all_framework_preprocessing:
                if (
                    y is not None
                    and isinstance(y, list)
                    and isinstance(y[0]["boxes"], np.ndarray)
                ):
                    y_tensor = []
                    for i, y_i in enumerate(y):
                        y_t = {}
                        y_t["boxes"] = (
                            torch.from_numpy(y_i["boxes"])
                            .type(torch.float)
                            .to(self.device)
                        )
                        y_t["labels"] = (
                            torch.from_numpy(y_i["labels"])
                            .type(torch.int64)
                            .to(self.device)
                        )
                        if "masks" in y_i:
                            y_t["masks"] = (
                                torch.from_numpy(y_i["masks"])
                                .type(torch.int64)
                                .to(self.device)
                            )
                        y_tensor.append(y_t)
                elif y is not None and isinstance(y, dict):
                    y_tensor = []
                    for i in range(y["boxes"].shape[0]):
                        y_t = {}
                        y_t["boxes"] = y["boxes"][i]
                        y_t["labels"] = y["labels"][i]
                        y_tensor.append(y_t)
                else:
                    y_tensor = y  # type: ignore

                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                )
                image_tensor_list_grad = []
                y_preprocessed = []
                inputs_t = []

                for i in range(x.shape[0]):
                    if isinstance(x, np.ndarray):
                        if self.clip_values is not None:
                            x_grad = transform(x[i] / self.clip_values[1]).to(
                                self.device
                            )
                        else:
                            x_grad = transform(x[i]).to(self.device)
                        x_grad.requires_grad = True
                    else:
                        x_grad = x[i].to(self.device)
                        if x_grad.shape[-1] in [1, 3, 6]:  # allows RGB + depth input
                            x_grad = torch.permute(x_grad, (2, 0, 1))

                    image_tensor_list_grad.append(x_grad)
                    x_grad_1 = torch.unsqueeze(x_grad, dim=0)
                    x_preprocessed_i, y_preprocessed_i = self._apply_preprocessing(
                        x_grad_1, y=[y_tensor[i]], fit=False, no_grad=False
                    )
                    for i_preprocessed in range(x_preprocessed_i.shape[0]):
                        inputs_t.append(x_preprocessed_i[i_preprocessed])
                        y_preprocessed.append(y_preprocessed_i[i_preprocessed])

            elif isinstance(x, np.ndarray):
                x_preprocessed, y_preprocessed = self._apply_preprocessing(
                    x, y=y, fit=False, no_grad=True
                )

                if y_preprocessed is not None and isinstance(
                    y_preprocessed[0]["boxes"], np.ndarray
                ):
                    y_preprocessed_tensor = []
                    for i, y_i in enumerate(y_preprocessed):
                        y_preprocessed_t = {}
                        y_preprocessed_t["boxes"] = (
                            torch.from_numpy(y_i["boxes"])
                            .type(torch.float)
                            .to(self.device)
                        )
                        y_preprocessed_t["labels"] = (
                            torch.from_numpy(y_i["labels"])
                            .type(torch.int64)
                            .to(self.device)
                        )
                        if "masks" in y_i:
                            y_preprocessed_t["masks"] = (
                                torch.from_numpy(y_i["masks"])
                                .type(torch.uint8)
                                .to(self.device)
                            )
                        y_preprocessed_tensor.append(y_preprocessed_t)
                    y_preprocessed = y_preprocessed_tensor

                transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                )
                image_tensor_list_grad = []

                for i in range(x_preprocessed.shape[0]):
                    if self.clip_values is not None:
                        x_grad = transform(x_preprocessed[i] / self.clip_values[1]).to(
                            self.device
                        )
                    else:
                        x_grad = transform(x_preprocessed[i]).to(self.device)
                    x_grad.requires_grad = True
                    image_tensor_list_grad.append(x_grad)

                inputs_t = image_tensor_list_grad

            else:
                raise NotImplementedError(
                    "Combination of inputs and preprocessing not supported."
                )

            if isinstance(y_preprocessed, np.ndarray):
                labels_t = torch.from_numpy(y_preprocessed).to(self.device)  # type: ignore
            else:
                labels_t = y_preprocessed  # type: ignore

            output = self._model(inputs_t, labels_t)

            return output, inputs_t, image_tensor_list_grad

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
                    self._patch[3:, :, :] = torch.clamp(
                        self._patch[3:, :, :], min=self.min_depth, max=self.max_depth
                    )
        else:
            raise ValueError(
                "Adam optimizer for CARLA Adversarial Patch not supported."
            )
            self._optimizer.step()

            with torch.no_grad():
                if self._patch.shape[0] == 6:
                    self._patch[3:, :, :] = torch.mean(
                        self._patch[3:, :, :], dim=0, keepdim=True
                    )
                    self._patch[3:, :, :] = torch.clamp(
                        self._patch[3:, :, :], min=self.min_depth, max=self.max_depth
                    )

                self._patch[:] = torch.clamp(
                    self._patch,
                    min=self.estimator.clip_values[0],
                    max=self.estimator.clip_values[1],
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

        def in_polygon(x, y, vertices):
            """
            Determine if a point (x,y) is inside a polygon with given vertices
            Ref: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
            """
            n_pts = len(vertices)
            i = 0
            j = n_pts - 1
            c = False

            while i < n_pts:
                if (
                    # y coordinate of the point has to be between the y coordinates of the i-th and j-th vertices
                    ((vertices[i][1] <= y) and (y < vertices[j][1]))
                    or ((vertices[j][1] <= y) and (y < vertices[i][1]))
                ) and (
                    # x coordinate of the point is to the left of the line connecting i-th and j-th vertices
                    x
                    < (vertices[j][0] - vertices[i][0])
                    * (y - vertices[i][1])
                    / (vertices[j][1] - vertices[i][1])
                    + vertices[i][0]
                ):
                    c = not c
                j = i
                i = i + 1
            return c

        def get_avg_depth_value(depth_img, patch_coords):
            # Return the average depth value of a patch with coordinates given by patch_coords
            avg_depth = 0.0
            count = 0
            for i in range(depth_img.shape[1]):
                for j in range(depth_img.shape[0]):
                    if in_polygon(i, j, patch_coords):
                        avg_depth = avg_depth + depth_img[j, i]
                        count = count + 1
            return avg_depth / count

        # Reference: https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py
        def linear_to_log(depth_meters):
            """
            Convert linear depth in meters to logorithmic depth between [0,1]
            """
            # Reference https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
            normalized_depth = depth_meters / 1000.0
            # Convert to logarithmic depth.
            depth_log = 1.0 + np.log(normalized_depth) / 5.70378
            depth_log = np.clip(depth_log, 0.0, 1.0)
            return depth_log

        def log_to_linear(depth_log):
            """
            Convert log depth between [0,1] to linear depth in meters
            """
            normalized_depth = np.exp((depth_log - 1.0) * 5.70378)
            depth_meters = normalized_depth * 1000.0
            return depth_meters

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
            patch_init = np.random.randint(0, 255, size=self.patch_shape) / 255

            if (
                self.patch_shape[0] == 6
            ):  # initialize depth patch with average depth value of green screen
                if "avg_patch_depth" in y_patch_metadata[i]:  # for Eval 5+ metadata
                    avg_patch_depth = y_patch_metadata[i]["avg_patch_depth"]
                else:  # backward compatible with Eval 4 metadata
                    avg_patch_depth = get_avg_depth_value(x[i][:, :, 3], gs_coords)
                avg_patch_depth_meters = log_to_linear(avg_patch_depth)
                max_depth_meters = avg_patch_depth_meters + self.depth_delta_meters
                min_depth_meters = avg_patch_depth_meters - self.depth_delta_meters
                self.max_depth = linear_to_log(max_depth_meters)
                self.min_depth = linear_to_log(min_depth_meters)
                patch_init[3:, :, :] = avg_patch_depth

            self._patch = torch.tensor(
                patch_init, requires_grad=True, device=self.estimator.device
            )

            log.info("y_gt: {}".format(y_gt))
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
