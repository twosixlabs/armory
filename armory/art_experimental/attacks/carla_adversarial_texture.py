from typing import Optional

from art.attacks.evasion import AdversarialTexturePyTorch
import numpy as np
import torch


class AdversarialPhysicalTexture(AdversarialTexturePyTorch):
    """
    ART's AdversarialTexturePytorch with overriden generate()
    """

    def __init__(self, estimator, **kwargs):
        # use dummy patch height/width for initialization
        super().__init__(estimator=estimator, patch_height=1, patch_width=1, **kwargs)

    # Copied from ART 1.10.0 and modified to accommodate dynamic shadowing on patch
    def _apply_texture(
        self,
        videos: "torch.Tensor",
        patch: "torch.Tensor",
        foreground: Optional["torch.Tensor"],
        patch_points: Optional[np.ndarray],
    ) -> "torch.Tensor":
        """
        Apply texture over background and overlay foreground.
        :param videos: Video samples.
        :param patch: Patch to apply.
        :param foreground: Foreground mask.
        :param patch_points: Array of shape (nb_frames, 4, 2) containing four pairs of integers (height, width)
                             corresponding to the coordinates of the four corners top-left, top-right, bottom-right,
                             bottom-left of the transformed image in the coordinate-system of the original image.
        :return: Patched videos.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        nb_samples = videos.shape[0]
        nb_frames = videos.shape[1]
        frame_height = videos.shape[2]
        frame_width = videos.shape[3]

        image_mask = self._get_patch_mask(nb_samples=nb_samples)
        image_mask = image_mask.float()

        patch = patch.float()
        padded_patch = torch.stack([patch] * nb_samples)

        if patch_points is None:
            pad_h_before = self.x_min
            pad_h_after = int(
                videos.shape[self.i_h + 1]
                - pad_h_before
                - image_mask.shape[self.i_h_patch + 1]
            )

            pad_w_before = self.y_min
            pad_w_after = int(
                videos.shape[self.i_w + 1]
                - pad_w_before
                - image_mask.shape[self.i_w_patch + 1]
            )

            image_mask = image_mask.permute(0, 3, 1, 2)

            image_mask = torchvision.transforms.functional.pad(
                img=image_mask,
                padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
                fill=0,
                padding_mode="constant",
            )

            image_mask = image_mask.permute(0, 2, 3, 1)

            image_mask = torch.unsqueeze(image_mask, dim=1)
            image_mask = torch.repeat_interleave(image_mask, dim=1, repeats=nb_frames)
            image_mask = image_mask.float()

            padded_patch = padded_patch.permute(0, 3, 1, 2)

            padded_patch = torchvision.transforms.functional.pad(
                img=padded_patch,
                padding=[pad_w_before, pad_h_before, pad_w_after, pad_h_after],
                fill=0,
                padding_mode="constant",
            )

            padded_patch = padded_patch.permute(0, 2, 3, 1)

            padded_patch = torch.unsqueeze(padded_patch, dim=1)
            padded_patch = torch.repeat_interleave(
                padded_patch, dim=1, repeats=nb_frames
            )

            padded_patch = padded_patch.float()

        else:

            startpoints = [
                [0, 0],
                [frame_width, 0],
                [frame_width, frame_height],
                [0, frame_height],
            ]
            endpoints = np.zeros_like(patch_points)
            endpoints[:, :, 0] = patch_points[:, :, 1]
            endpoints[:, :, 1] = patch_points[:, :, 0]

            image_mask = image_mask.permute(0, 3, 1, 2)

            image_mask = torchvision.transforms.functional.resize(
                img=image_mask,
                size=[int(videos.shape[2]), int(videos.shape[3])],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            image_mask_list = []

            for i_frame in range(nb_frames):

                image_mask_i = torchvision.transforms.functional.perspective(
                    img=image_mask,
                    startpoints=startpoints,
                    endpoints=endpoints[i_frame],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )

                image_mask_i = image_mask_i.permute(0, 2, 3, 1)

                image_mask_list.append(image_mask_i)

            image_mask = torch.stack(image_mask_list, dim=1)
            image_mask = image_mask.float()

            padded_patch = padded_patch.permute(0, 3, 1, 2)

            padded_patch = torchvision.transforms.functional.resize(
                img=padded_patch,
                size=[int(videos.shape[2]), int(videos.shape[3])],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            padded_patch_list = []

            for i_frame in range(nb_frames):
                padded_patch_i = torchvision.transforms.functional.perspective(
                    img=padded_patch,
                    startpoints=startpoints,
                    endpoints=endpoints[i_frame],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )

                padded_patch_i = padded_patch_i.permute(0, 2, 3, 1)

                padded_patch_list.append(padded_patch_i)

            padded_patch = torch.stack(padded_patch_list, dim=1)
            padded_patch = padded_patch.float()

        inverted_mask = (
            torch.from_numpy(np.ones(shape=image_mask.shape, dtype=np.float32)).to(
                self.estimator.device
            )
            - image_mask
        )

        # Adjust green screen brightness
        v_avg = (
            0.5647  # average V value (in HSV) for the green screen, which is #00903a
        )
        green_screen = videos * image_mask
        values, _ = torch.max(green_screen, dim=4, keepdim=True)
        values_ratio = values / v_avg
        values_ratio = torch.repeat_interleave(values_ratio, dim=4, repeats=3)

        if foreground is not None:
            combined = (
                videos * inverted_mask
                + padded_patch * values_ratio * image_mask
                - padded_patch * values_ratio * ~foreground.bool()
                + videos * ~foreground.bool() * image_mask
            )

            combined = torch.clamp(
                combined,
                min=self.estimator.clip_values[0],
                max=self.estimator.clip_values[1],
            )
        else:
            combined = videos * inverted_mask + padded_patch * values_ratio * image_mask

        return combined

    def generate(self, x, y, y_patch_metadata=None, **kwargs):
        """
        :param x: Sample videos with shape (NFHWC)
        :param y: True labels of format `List[Dict[str, np.ndarray]]`, one dictionary for each input video. The keys of
                  the dictionary are:
                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                         0 <= y1 < y2 <= H.
        :param y_patch_metadata: Metadata of the green screen patch of format `List[Dict[str, np.ndarray]]`. The keys of
                  the dictionary are:
                  - gs_coords: the coordinates of the patch in [top_left, top_right, bottom_right, bottom_left] format
                  - cc_ground_truth: ground truth color information stored as np.ndarray with shape (24,3)
                  - cc_scene: scene color information stored as np.ndarray with shape (24,3)
                  - masks: binarized masks of the patch, where masks[n,x,y] == 1 means patch pixel in frame n and at position (x,y)
        """

        if x.shape[0] > 1:
            raise ValueError("batch size must be 1")

        # green screen coordinates used for placement of a rectangular patch
        gs_coords = y_patch_metadata[0]["gs_coords"]
        if gs_coords.ndim == 2:  # same location for all frames
            patch_width = int(np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0]))
            patch_height = int(np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1]))
        else:
            patch_widths = []
            patch_heights = []
            for coords in gs_coords:
                patch_widths.append(int(np.max(coords[:, 0]) - np.min(coords[:, 0])))
                patch_heights.append(int(np.max(coords[:, 1]) - np.min(coords[:, 1])))
            patch_width = max(patch_widths)
            patch_height = max(patch_heights)

        self.patch_height = patch_height
        self.patch_width = patch_width

        # reinitialize patch
        self.patch_shape = (patch_height, patch_width, 3)
        mean_value = (
            self.estimator.clip_values[1] - self.estimator.clip_values[0]
        ) / 2.0 + self.estimator.clip_values[0]
        self._initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = torch.tensor(
            self._initial_value, requires_grad=True, device=self.estimator.device
        )

        # this masked to embed patch into the background in the event of occlusion
        foreground = y_patch_metadata[0]["masks"]
        foreground = np.array([foreground])

        # create patch points indicating locations of the four corners of the patch in each frame
        if gs_coords.ndim == 2:  # same location for all frames
            patch_points = np.tile(gs_coords[:, ::-1], (x.shape[1], 1, 1))
        else:
            patch_points = gs_coords[:, :, ::-1]

        generate_kwargs = {
            "y_init": y[0]["boxes"][0:1],
            "foreground": foreground,
            "shuffle": kwargs.get("shuffle", False),
            "patch_points": patch_points,
        }
        generate_kwargs = {**generate_kwargs, **kwargs}

        attacked_video = super().generate(x, y, **generate_kwargs)

        return attacked_video
