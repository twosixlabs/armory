import numpy as np
import torch

from art.attacks.evasion import AdversarialTexturePyTorch


class AdversarialPhysicalTexture(AdversarialTexturePyTorch):
    """
    ART's AdversarialTexturePytorch with overriden generate()
    """

    def __init__(self, estimator, **kwargs):
        # self.attack_kwargs = kwargs
        # super(AdversarialTexturePyTorch, self).__init__(estimator=estimator)

        # use dummy patch height/width for initialization
        super().__init__(estimator=estimator, patch_height=1, patch_width=1, **kwargs)

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
        patch_width = int(np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0]))
        patch_height = int(np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1]))

        x_min = int(np.min(gs_coords[:, 1]))
        y_min = int(np.min(gs_coords[:, 0]))

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.x_min = x_min
        self.y_min = y_min

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
        patch_points = []
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
