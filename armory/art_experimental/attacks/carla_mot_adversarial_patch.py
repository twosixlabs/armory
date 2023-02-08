import os
from typing import Optional, Tuple

from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
import cv2
import numpy as np
import torch
from tqdm import trange

from armory.logs import log


class CARLAMOTAdversarialPatchPyTorch(AdversarialPatchPyTorch):
    def __init__(self, estimator, coco_format=False, **kwargs):

        self.batch_frame_size = kwargs.pop(
            "batch_frame_size", 1
        )  # number of frames to attack per iteration
        self.patch_base_image = kwargs.pop("patch_base_image", None)
        self.coco_format = bool(coco_format)

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
            endpoints = self.patch_coords_batch[
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
            torch.tensor(self.patch_masks_batch == 0), dim=-1, keepdim=True
        ).to(self.estimator.device)
        foreground_mask = torch.permute(foreground_mask, (0, 3, 1, 2))

        # Adjust green screen brightness
        v_avg = (
            0.5647  # average V value (in HSV) for the green screen, which is #00903a
        )
        green_screen = images * image_mask
        values, _ = torch.max(green_screen[:, :3, :, :], dim=1, keepdim=True)
        values_ratio = values / v_avg
        values_ratio = torch.repeat_interleave(values_ratio, dim=1, repeats=3)

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

    def inner_generate(  # type: ignore
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dervied from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/1.12.0/art/attacks/evasion/adversarial_patch/adversarial_patch_pytorch.py,
        we eliminate the PyTorch dataloader, due to its default inability to handle batching of variable size bounding box annotations, and
        we eliminate code not relevant to the CARLA MOT datasets.

        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NCHW or input videos of shape NFCHW.
        :param y: An array with the original true labels.
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import torch  # lgtm [py/repeated-import]

        for i_iter in trange(
            self.max_iter, desc="Adversarial Patch PyTorch", disable=not self.verbose
        ):
            images = torch.from_numpy(x).to(self.estimator.device)
            _ = self._train_step(images=images, target=y, mask=None)

            # Write summary
            if self.summary_writer is not None:  # pragma: no cover
                x_patched = (
                    self._random_overlay(
                        images=torch.from_numpy(x).to(self.estimator.device),
                        patch=self._patch,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_iter,
                    grad=None,
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y,
                    targeted=self.targeted,
                )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return (
            self._patch.detach().cpu().numpy(),
            self._get_circular_patch_mask(nb_samples=1).cpu().numpy()[0],
        )

    # can probably move to art_experimental/attacks/carla_obj_det_utils.py
    @staticmethod
    def create_art_annotations_from_mot(y):
        """
        param y: 2D NDArray of shape (M, 9), where M is the total number of detections and each detection has format:
                <timestep> <object_id> <bbox top-left x> <bbox top-left y> <bbox width> <bbox height> <confidence_score=1> <class_id> <visibility=1>
        return y_art: list of dict, where each dict represents ART-style annotations
            of each frame in the video

        NOTE: This is NOT true coco format: https://cocodataset.org/#format-results
            The fields are ART-specific: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/1.12.1/art/attacks/evasion/adversarial_patch/adversarial_patch_pytorch.py#L550-L552
        """
        y_art = []
        n_timesteps = int(np.max(y[:, 0])) + 1
        for t in range(n_timesteps):
            # convert from (x,y,w,h) to (x1,y1,x2,y2)
            boxes = y[y[:, 0] == t, 2:6]  # <timestep> is 0-based
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            scores = y[y[:, 0] == t, 6]
            labels = y[y[:, 0] == t, 7]
            y_art.append(
                {
                    "boxes": np.array(boxes, dtype=np.float32),
                    "labels": np.array(labels, dtype=np.float32),
                    "scores": np.array(scores, dtype=np.float32),
                }
            )
        return y_art

    @staticmethod
    def create_art_annotations_from_coco(y):
        """
        param y: coco format data for a single video
            See: armory.data.adversarial_datasets.mot_array_to_coco
        return y_art: list of dict, where each dict represents ART-style annotations
            of each frame in the video

        NOTE: This is NOT true coco format: https://cocodataset.org/#format-results
            The fields are ART-specific: https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/1.12.1/art/attacks/evasion/adversarial_patch/adversarial_patch_pytorch.py#L550-L552
        """
        y_art = []
        n_timesteps = max(y_i["image_id"] for y_i in y) + 1
        for t in range(n_timesteps):
            boxes, labels, scores = [], [], []
            y_t = [y_i for y_i in y if y_i["image_id"] == t]
            for y_i in y_t:
                # convert from (x,y,w,h) to (x1,y1,x2,y2)
                x1, y1, w, h = y_i["bbox"]
                boxes.append([x1, y1, x1 + w, y1 + h])
                scores.append(y_i["score"])
                labels.append(y_i["category_id"])
            y_art.append(
                {
                    "boxes": np.array(boxes, dtype=np.float32),
                    "labels": np.array(labels, dtype=np.float32),
                    "scores": np.array(scores, dtype=np.float32),
                }
            )
        return y_art

    def generate(self, x, y, y_patch_metadata):
        """
        param x: Sample video of shape=(NFHWC).
        param y: Sample labels of shape (NM9), where M is the total number of detections in a single video
            ith dictionary contains bounding boxes, class labels, and class scores
        param y_patch_metadata: Patch metadata. List of N dictionaries, ith dictionary contains patch metadata for x[i]
        """

        if x.shape[0] > 1:
            log.info("To perform per-example patch attack, batch size must be 1")
        assert x.shape[-1] in [3], "x must have 3 channels"

        num_vids = x.shape[0]  # shape = (NFHWC)
        attacked_videos = []
        for i in range(num_vids):
            # Adversarial patch attack, when used for object detection, requires ground truth
            if self.coco_format:
                y_gt_coco = self.create_art_annotations_from_coco(y[i])
            else:
                y_gt_coco = self.create_art_annotations_from_mot(y[i])
            self.gs_coords = y_patch_metadata[i]["gs_coords"]  # patch coordinates
            patch_width = np.max(self.gs_coords[:, :, 0]) - np.min(
                self.gs_coords[:, :, 0]
            )
            patch_height = np.max(self.gs_coords[:, :, 1]) - np.min(
                self.gs_coords[:, :, 1]
            )
            self.patch_shape = (
                x.shape[-1],
                patch_height,
                patch_width,
            )

            # Use this mask to embed patch into the background in the event of occlusion
            self.patch_masks_video = y_patch_metadata[i]["masks"]

            # self._patch needs to be re-initialized with the correct shape
            if self.patch_base_image is not None:
                self.patch_base = self.create_initial_image(
                    (patch_width, patch_height),
                )
                patch_init = self.patch_base
            else:
                patch_init = np.random.randint(0, 255, size=self.patch_shape) / 255

            self._patch = torch.tensor(
                patch_init, requires_grad=True, device=self.estimator.device
            )

            # Perform batch attack by attacking multiple frames from the same video
            for batch_i in range(0, x[i].shape[0], self.batch_frame_size):
                batch_i_end = min(batch_i + self.batch_frame_size, x[i].shape[0])
                self.patch_masks_batch = self.patch_masks_video[
                    batch_i:batch_i_end, :, :, :
                ]
                self.patch_coords_batch = self.gs_coords[batch_i:batch_i_end, :, :]
                x_batch = x[i][batch_i:batch_i_end, :, :, :]
                y_batch = y_gt_coco[batch_i:batch_i_end]
                self.inner_generate(x_batch, y=y_batch)

            # Patch image
            x_tensor = torch.tensor(x[i]).to(self.estimator.device)
            self.patch_masks_batch = self.patch_masks_video
            self.patch_coords_batch = self.gs_coords
            patched_image = (
                self._random_overlay(
                    images=x_tensor, patch=self._patch, scale=None, mask=None
                )
                .detach()
                .cpu()
                .numpy()
            )

            # Embed patch into background
            patched_image[np.all(self.patch_masks_video == 0, axis=-1)] = x[i][
                np.all(self.patch_masks_video == 0, axis=-1)
            ]

            patched_image = np.clip(
                patched_image,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            attacked_videos.append(patched_image)

        return np.array(attacked_videos)
