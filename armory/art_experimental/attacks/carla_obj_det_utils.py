from dataclasses import dataclass
import inspect
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from armory.logs import log
from armory.utils.shape_gen import Shape


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
    depth_meters: scalar or array, nonnegative
    returns: scalar or array in [0,1]
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
    depth_log: scalar or array in [0,1]
    returns: scalar or array, nonnegative
    """
    normalized_depth = np.exp((depth_log - 1.0) * 5.70378)
    depth_meters = normalized_depth * 1000.0
    return depth_meters


def linear_depth_to_rgb(depth_m):
    """
    Converts linear depth in meters to RGB values between [0,1]
    Reference: https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map
    depth_m: scalar or array, nonnegative
    returns: tuple of three scalars or arrays in [0,1]
    """
    depth = depth_m / 1000.0 * (256**3 - 1)
    if isinstance(depth, np.ndarray):
        depth = np.round(depth)
    elif torch.is_tensor(depth):
        depth = torch.round(depth)
    else:
        depth = round(depth)

    r = depth % 256
    g = ((depth - r) / 256.0) % 256
    b = (depth - r - g * 256) / 256**2
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    return r, g, b


def rgb_depth_to_linear(r, g, b):
    """
    Converts rgb depth values between [0,1] to linear depth in meters.
    r, g, b: three scalars or arrays with the same shape in [0,1]
    returns: scalar or array, nonnegative
    """
    r_ = r * 255.0
    g_ = g * 255.0
    b_ = b * 255.0
    depth_m = r_ + g_ * 256 + b_ * 256 * 256
    depth_m = depth_m * 1000.0 / (256**3 - 1)
    return depth_m


@dataclass
class PatchMask:
    """A patch mask for a single image.
    Fields:
        path: Path to the mask image. Must be in the same directory as attack code.
        shape: Optionally supply valid shape name from armory.utils.shape_gen.SHAPES to generate a shape.
            Will ignore path if both are supplied.
        invert: Whether to invert the mask. Defaults to True.
        fill: How to fill the masked area, one of:
            - "init": Fill with the patch_initialization values.
            - "random": Fill with random values.
            - color: Fill with a single color, specified with hex RGB values (0xRRGGBB).
            - path: Fill with an image specified by path. Must be in the same directory as attack code.
    """

    path: Optional[str]
    shape: Optional[Shape]
    invert: bool
    fill: str

    @classmethod
    def from_kwargs(cls, kwargs) -> Optional["PatchMask"]:
        mask = kwargs.pop("patch_mask", None)
        if isinstance(mask, str):
            return PatchMask(path=mask, shape=None, invert=True, fill="init")
        elif isinstance(mask, dict):
            return PatchMask(
                path=mask.get("path", None),
                shape=Shape.from_name(
                    mask.get("shape", None), mask.get("shape_kwargs", None)
                ),
                invert=mask.get("invert", True),
                fill=mask.get("fill", "init"),
            )
        elif mask is None:
            return None
        raise ValueError(f"patch_mask must be a string or dictionary, got: {mask}")

    @staticmethod
    def _path_search(path, stack_offset=1):
        if os.path.exists(path):
            return path
        # Get call stack filepaths, removing duplicates and non-files
        call_stack = [
            *dict.fromkeys(
                [x.filename for x in inspect.stack() if os.path.exists(x.filename)]
            )
        ]
        calling_module = call_stack[stack_offset]
        calling_path = os.path.abspath(
            os.path.join(os.path.dirname(calling_module), path)
        )
        if os.path.exists(calling_path):
            return calling_path
        cur_module = call_stack[0]
        cur_path = os.path.abspath(os.path.join(os.path.dirname(cur_module), path))
        if os.path.exists(cur_path):
            return cur_path
        raise ValueError(
            f"Could not find mask image {path}. Must be in same dir as {calling_module} or {cur_module}."
        )

    def __post_init__(self):
        if self.shape is not None:
            if self.path is not None:
                log.warning("Ignoring patch_mask.path because patch_mask.shape is set.")
                self.path = None
        else:
            self.path = self._path_search(self.path, stack_offset=1)

    def _load(self) -> np.ndarray:
        """Load the mask image."""
        if self.shape is not None:
            mask = self.shape.array
        else:
            mask = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        _src = self.shape.name if self.shape is not None else self.path
        if mask is None:
            raise ValueError(f"Could not load mask image {_src}")
        if np.all(mask == 255):
            raise ValueError(
                f"Mask image {_src} is all white, transparent pixels are treated as white."
            )
        return mask

    @staticmethod
    def project_mask(mask, target_shape, gs_coords, as_bool=True) -> np.ndarray:
        """Project a mask onto target greenscreen coordinates."""
        # Define the tgt points for the perspective transform
        dst_pts = gs_coords.astype(np.float32)

        # Define the source points for the perspective transform
        src_pts = np.array(
            [
                [0, 0],
                [target_shape[1], 0],
                [target_shape[1], target_shape[0]],
                [0, target_shape[0]],
            ],
            np.float32,
        )

        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective transformation to the mask tensor
        resized_mask = cv2.resize(
            mask[:, :, :3], dsize=target_shape[:2][::-1], interpolation=cv2.INTER_LINEAR
        )
        mask_transformed = cv2.warpPerspective(resized_mask, M, target_shape[:2][::-1])

        if as_bool:
            return np.where(mask_transformed == 0, False, True)

        return mask_transformed

    def project(
        self,
        shape: Tuple[int, int, int],
        gs_coords: np.ndarray,
        as_bool: bool = True,
        mask: Optional[np.ndarray] = None,
    ):
        """Project the mask onto an image of the given shape."""
        if mask is None:
            mask = self._load()
        proj = self.project_mask(mask, shape, gs_coords, as_bool=as_bool)
        # remove whitespace and transparency
        proj = np.where(proj.all(axis=-1), 0, proj.any(axis=-1)).astype(mask.dtype)
        if proj.dtype == np.uint8:
            proj = proj * 255
        # add back channels
        proj = np.stack([proj] * 3, axis=-1)
        if as_bool:
            proj = proj.astype(bool)
        if self.invert:
            proj = ~proj
        return proj

    def fill_masked_region(
        self, patched_image, projected_mask, gs_coords, patch_init, orig_patch_mask
    ):
        masked_region = self.project_mask(
            np.ones((1, 1, 3), dtype=np.uint8) * 255, patched_image.shape, gs_coords
        )
        boolean_mask = ~projected_mask * masked_region
        boolean_mask = np.prod(boolean_mask, axis=-1, keepdims=True).astype(bool)
        if self.fill == "init":
            fill = patch_init
        elif self.fill == "random":
            fill = np.random.randint(0, 255, size=self.patch_shape) / 255
        elif self.fill.upper().startswith("0X"):
            fill = np.array(
                [
                    int(self.fill[2:4], 16),
                    int(self.fill[4:6], 16),
                    int(self.fill[6:8], 16),
                ]
            )
            if any(fill == 0):
                log.warning(
                    "Patch mask fill color a contains 0 in RGB. Setting to 1 instead."
                )
            fill[fill == 0] = 1  # hack
            fill = np.ones_like(patch_init) * fill[:, np.newaxis, np.newaxis] / 255
        elif os.path.isfile(valid_path := self._path_search(self.fill)):
            self.fill = valid_path
            fill = cv2.imread(self.fill, cv2.IMREAD_UNCHANGED)
            patch_width = np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0])
            patch_height = np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1])
            fill = cv2.resize(fill, (patch_width, patch_height))
            if (fill == 0).any():
                log.warning(
                    "Patch mask fill color a contains 0 in RGB. Setting to 1 instead."
                )
            fill[fill == 0] = 1  # hack
            fill = np.transpose(fill, (2, 0, 1)).astype(patched_image.dtype) / 255
        else:
            raise ValueError(
                f"Invalid patch mask fill value. Must be 'init', 'random', '0xRRGGBB', or a valid filepath."
                f" Got {self.fill}"
            )
        fill = np.transpose(fill, (1, 2, 0)).astype(patched_image.dtype)
        projected_init = self.project_mask(
            fill, projected_mask.shape, gs_coords, as_bool=False
        )
        masked_init = projected_init * orig_patch_mask.astype(bool) * boolean_mask
        gs_mask = masked_init.astype(bool)
        if np.logical_xor(gs_mask[::, :2], gs_mask[::, -2:]).sum() != 0:
            log.warning("gs_mask is not reducable")
        gs_mask = np.prod(gs_mask, axis=-1).astype(bool)
        gs_mask = np.expand_dims(gs_mask, axis=-1).repeat(
            patched_image.shape[-1], axis=-1
        )
        patched_image_inverse_mask = patched_image * ~gs_mask
        if masked_init.shape != patched_image_inverse_mask.shape:
            # Depth channels are missing
            log.warning("masked_init is missing depth channels. Adding them back in.")
            assert (
                patched_image_inverse_mask.shape[:-1] == masked_init.shape[:-1]
                and masked_init.shape[-1] * 2 == patched_image_inverse_mask.shape[-1]
            ), "masked_init has an invalid shape"
            masked_init = np.concatenate(
                (masked_init, patched_image[:, :, 3:6] * gs_mask[:, :, :3]), axis=-1
            )
            assert masked_init.shape == patched_image_inverse_mask.shape
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        # ax[0][0].imshow(patched_image[:,:,:3])
        # ax[0][0].set_title("masked adversarial perturbation")
        # ax[0][1].imshow(projected_init[:,:,:3])
        # ax[0][1].set_title("projected initialization image")
        # ax[0][2].imshow(gs_mask.astype(float)[:,:,:3])
        # ax[0][2].set_title("init image embedding mask")
        # ax[1][0].imshow(masked_init[:,:,:3])
        # ax[1][0].set_title("init embedding")
        # ax[1][1].imshow(patched_image_inverse_mask[:,:,:3])
        # ax[1][1].set_title("patched image w/o embedding")
        # ax[1][2].imshow(patched_image_inverse_mask[:,:,:3] + masked_init[:,:,:3])
        # ax[1][2].set_title("patched image w/ embedding")
        # plt.show()
        # breakpoint()
        return patched_image_inverse_mask + masked_init
