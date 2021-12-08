import logging
import numpy as np

from art.attacks.evasion import AdversarialTexturePyTorch

logger = logging.getLogger(__name__)


class AdversarialPhysicalTexture(AdversarialTexturePyTorch):
    """
    ART's AdversarialTexturePytorch with overriden generate()
    """

    def __init__(self, estimator, **kwargs):
        self.attack_kwargs = kwargs
        super(AdversarialTexturePyTorch, self).__init__(estimator=estimator)

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
        :Keyword Arguments:
            * *shuffle* (``np.ndarray``) --
              Shuffle order of samples, labels, initial boxes, and foregrounds for texture generation.
            * *y_init* (``np.ndarray``) --
              Initial boxes around object to be tracked of shape (nb_samples, 4) with second dimension representing
              [x1, y1, x2, y2] with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
            * *foreground* (``np.ndarray``) --
              Foreground masks of shape NFHWC of boolean values with False/0.0 representing foreground, preventing
              updates to the texture, and True/1.0 for background, allowing updates to the texture.
        :return: An array with adversarial patch and an array of the patch mask.
        """

        if x.shape[0] > 1:
            raise ValueError("batch size must be 1")

        # green screen coordinates used for placement of a rectangular patch
        gs_coords = y_patch_metadata[0]["gs_coords"]
        patch_width = int(np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0]))
        patch_height = int(np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1]))

        x_min = int(np.min(gs_coords[:, 1]))
        y_min = int(np.min(gs_coords[:, 0]))

        attack = AdversarialTexturePyTorch(
            self.estimator,
            patch_height=patch_height,
            patch_width=patch_width,
            x_min=x_min,
            y_min=y_min,
            **self.attack_kwargs
        )

        # this masked to embed patch into the background in the event of occlusion
        foreground = y_patch_metadata[0]["masks"]
        foreground = np.array([foreground])

        generate_kwargs = {
            "y_init": y[0]["boxes"][0:1],
            "foreground": foreground,
            "shuffle": kwargs.get("shuffle", False),
        }
        generate_kwargs = {**generate_kwargs, **kwargs}
        attacked_video = attack.generate(x, y, **generate_kwargs)
        return attacked_video
