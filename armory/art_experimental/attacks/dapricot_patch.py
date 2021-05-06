"""
Copyright 2021 The MITRE Corporation. All rights reserved
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
import cv2
import math
import colour
from scipy.stats import norm
import random

from art.attacks.evasion import RobustDPatch, ProjectedGradientDescent
from art import config

logger = logging.getLogger(__name__)


class RobustDPatchTargeted(RobustDPatch):
    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        patch_location: Tuple[int, int] = (0, 0),
        crop_range: Tuple[int, int] = (0, 0),
        brightness_range: Tuple[float, float] = (1.0, 1.0),
        rotation_weights: Union[
            Tuple[float, float, float, float], Tuple[int, int, int, int]
        ] = (1, 0, 0, 0),
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = True,
        verbose: bool = True,
    ):
        super(RobustDPatch, self).__init__(estimator=estimator)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE)
        else:
            self._patch = (
                np.random.randint(0, 255, size=patch_shape)
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            ).astype(config.ART_NUMPY_DTYPE)
        self.verbose = verbose
        self.patch_location = patch_location
        self.crop_range = crop_range
        self.brightness_range = brightness_range
        self.rotation_weights = rotation_weights
        self.sample_size = sample_size
        self._targeted = targeted
        self._check_params()

    def generate(
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate RobustDPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
        channel_index = x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError(
                "The color channel index of the images and the patch have to be identical."
            )
        if y is None and self.targeted:
            raise ValueError(
                "The targeted version of RobustDPatch attack requires target labels provided to `y`."
            )
        if y is not None and not self.targeted:
            raise ValueError("The RobustDPatch attack does not use target labels.")
        if x.ndim != 4:
            raise ValueError("The adversarial patch can only be applied to images.")

        image_height, image_width = x.shape[1:3]

        if y is not None:
            for i_image in range(x.shape[0]):
                y_i = y[i_image]["boxes"]
                for i_box in range(y_i.shape[0]):
                    x_1, y_1, x_2, y_2 = y_i[i_box]
                    if (
                        x_1 < self.crop_range[1]
                        or y_1 < self.crop_range[0]
                        or x_2 > image_width - self.crop_range[1] + 1
                        or y_2 > image_height - self.crop_range[0] + 1
                    ):
                        raise ValueError(
                            "Cropping is intersecting with at least one box, reduce `crop_range`."
                        )

        if (
            self.patch_location[0] + self.patch_shape[0]
            > image_height - self.crop_range[0]
            or self.patch_location[1] + self.patch_shape[1]
            > image_width - self.crop_range[1]
        ):
            raise ValueError("The patch (partially) lies outside the cropped image.")

        for i_step in trange(
            self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose
        ):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)

            for e_step in range(self.sample_size):
                if e_step == 0 or (e_step + 1) % 100 == 0:
                    logger.info("EOT Step: %i", e_step + 1)

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    if y is None:
                        y_batch = y
                    else:
                        y_batch = y[i_batch_start:i_batch_end]

                    # Sample and apply the random transformations:
                    (
                        patched_images,
                        patch_target,
                        transforms,
                    ) = self._augment_images_with_patch(
                        x[i_batch_start:i_batch_end],
                        y_batch,
                        self._patch,
                        channels_first=self.estimator.channels_first,
                    )

                    gradients = self.estimator.loss_gradient(
                        x=patched_images, y=patch_target,
                    )

                    gradients = self._untransform_gradients(
                        gradients,
                        transforms,
                        channels_first=self.estimator.channels_first,
                    )

                    patch_gradients = patch_gradients_old + np.sum(gradients, axis=0)
                    logger.debug(
                        "Gradient percentage diff: %f)",
                        np.mean(
                            np.sign(patch_gradients) != np.sign(patch_gradients_old)
                        ),
                    )

                    patch_gradients_old = patch_gradients

            self._patch = (
                self._patch
                + np.sign(patch_gradients)
                * (1 - 2 * int(self.targeted))
                * self.learning_rate
            )

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch,
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                )

        return self._patch


def shape_coords(h, w, obj_shape):
    if obj_shape == "octagon":
        smallest_side = min(h, w)
        pi = math.pi
        h_init = 0
        w_init = smallest_side / 2 * math.sin(pi / 2) / math.sin(3 * pi / 8)
        rads = np.linspace(0, -2 * pi, 8, endpoint=False) + 5 * pi / 8
        h_coords = [h_init * math.cos(r) - w_init * math.sin(r) + h / 2 for r in rads]
        w_coords = [h_init * math.sin(r) + w_init * math.cos(r) + w / 2 for r in rads]
        coords = [
            (int(round(wc)), int(round(hc))) for hc, wc in zip(h_coords, w_coords)
        ]
    elif obj_shape == "diamond":
        if h <= w:
            coords = np.array(
                [
                    [w // 2, 0],
                    [min(w - 1, int(round(w / 2 + h / 2))), h // 2],
                    [w // 2, h - 1],
                    [max(0, int(round(w / 2 - h / 2))), h // 2],
                ]
            )
        else:
            coords = np.array(
                [
                    [w // 2, max(0, int(round(h / 2 - w / 2)))],
                    [w - 1, h // 2],
                    [w // 2, min(h - 1, int(round(h / 2 + w / 2)))],
                    [0, h // 2],
                ]
            )
    elif obj_shape == "rect":
        # 0.8 w/h aspect ratio
        if h > w:
            coords = np.array(
                [
                    [max(0, int(round(w / 2 - 0.4 * h))), 0],
                    [min(w - 1, int(round(w / 2 + 0.4 * h))), 0],
                    [min(w - 1, int(round(w / 2 + 0.4 * h))), h - 1],
                    [max(0, int(round(w / 2 - 0.4 * h))), h - 1],
                ]
            )
        else:
            coords = np.array(
                [
                    [0, max(0, int(round(h / 2 - w / 1.6)))],
                    [w - 1, max(0, int(round(h / 2 - w / 1.6)))],
                    [w - 1, min(h - 1, int(round(h / 2 + w / 1.6)))],
                    [0, min(h - 1, int(round(h / 2 + w / 1.6)))],
                ]
            )
    else:
        raise ValueError('obj_shape can only be {"rect", "diamond", "octagon"}')

    return np.array(coords)


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


def create_mask(mask_type, h, w):
    """
    create mask according to shape
    """

    mask = np.zeros((h, w, 3))
    coords = shape_coords(h, w, mask_type)

    for i in range(h):
        for j in range(w):
            if in_polygon(i, j, coords):
                mask[i, j, :] = 1

    return mask


def calculate_ccm(im_np, gt_np, gamma=2.2, Vandermonde=True, degree=1):
    """
    Calculates the color transform matrix.

    attributes::
        im_np
            np array of color patch values from image
        gt_np
            np array of known color patch values
        gamma
            default is 2.2, this is most common
        Vandermonde
            boolean indicating whether to use basic ccm method or
            Vandermonde method
        degree
            default is 1, this is only used with Vandermonde method

    returns::
        color correction matrix
    """
    # normalize both arrays
    im_np = im_np / 255
    gt_np = gt_np / 255

    # linearize values by decoding gamma from RGBs
    im_lin = np.power(im_np, gamma)
    gt_lin = np.power(gt_np, gamma)

    # calculate matrix
    if Vandermonde:
        ccm = colour.characterisation.colour_correction_matrix_Vandermonde(
            gt_lin, im_lin
        )
    else:
        ccm = np.linalg.pinv(np.asmatrix(gt_lin)).dot(np.asmatrix(im_lin))

    return ccm


def apply_ccm(patch, ccm, gamma=2.2, Vandermonde=True, degree=1):
    """
    Applies transform to patch.

    attributes::
        patch
            np array of patch to be inserted
        ccm
            color correction matrix
        gamma
            default is still 2.2
        Vandermonde
            boolean indicating whether basic or Vandermonde method is
            being used for calculations
        degree
            default is 1, should only be used when Vandermonde is True

    returns::
        color transformed patch
    """
    # normalize image
    patch = patch / 255

    # linearize image
    patch_lin = np.power(patch, gamma)

    # get shape of patch
    rows, cols, ch = patch_lin.shape

    if Vandermonde:
        # reshape for matrix multiplication
        patch_reshape = np.reshape(patch_lin, (-1, 3))

        # expand patch for transform
        patch_expand = colour.characterisation.polynomial_expansion_Vandermonde(
            patch_reshape, degree
        )

        # multiply and reshape
        corrected_RGB = np.reshape(
            np.transpose(np.dot(ccm, np.transpose(patch_expand))), (rows, cols, ch)
        )
    else:
        # reshape for matrix multiplication
        patch_reshape = np.reshape(patch_lin, (rows * cols, ch))

        # multiply
        corrected_RGB = np.matmul(np.asmatrix(patch_reshape), ccm)

        # reshape back to normal
        corrected_RGB = np.reshape(np.array(corrected_RGB), (rows, cols, ch))

    # clip where necessary
    corrected_RGB = np.array(corrected_RGB)
    corrected_RGB[corrected_RGB > 1.0] = 1.0
    corrected_RGB[corrected_RGB < 0.0] = 0.0

    # reapply gamma
    corrected_RGB = np.power(corrected_RGB, (1 / gamma))

    # compensate for saturated pixels
    corrected_RGB[patch_lin == 1.0] = 1.0

    return corrected_RGB


def find_blur(image, patch, coords=[]):
    """
    Determines blur of the patch area.

    attributes::
        image
            real image with green screen inserted
        patch
            patch to be blurred
    """
    # convert image to grayscale and normalize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_chip = gray[coords[0, 1] : coords[1, 1], coords[0, 0]].astype("float32")
    gray_norm = gray_chip / 255.0

    # differentiate to estimate lsf
    lsf = np.diff(gray_norm)
    lsf[lsf < 0.0] = 0.0
    norm_lsf = lsf / np.max(lsf)

    # fit gaussian to lsf data
    _, std = norm.fit(norm_lsf)

    # determine kernel scaling
    size = math.ceil(3 * std)
    if size % 2 == 0:
        size += 1

    # blur patch
    blurred_im = cv2.GaussianBlur(patch, (size, size), std)

    return blurred_im


def insert_transformed_patch(patch, image, gs_shape, patch_coords=[], image_coords=[]):
    """
    Insert patch to image based on given or selected coordinates

    attributes::
        patch
            patch as numpy array
        image
            image as numpy array
        gs_shape
            green screen shape
        patch_coords
            patch coordinates to map to image [numpy array]
        image_coords
            image coordinates patch coordinates will be mapped to [numpy array]
            going in clockwise direction, starting with upper left corner

    returns::
        image with patch inserted
    """

    # if no patch coords are given, just use whole image
    if patch_coords == []:
        patch_coords = shape_coords(patch.shape[0], patch.shape[1], gs_shape)

    # calculate homography
    h, status = cv2.findHomography(patch_coords, image_coords)

    # mask to aid with insertion
    mask = create_mask(gs_shape, patch.shape[0], patch.shape[1])
    mask_out = cv2.warpPerspective(
        mask, h, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC
    )

    # mask patch and warp it to destination coordinates
    patch[mask == 0] = 0
    im_out = cv2.warpPerspective(
        patch, h, (image.shape[1], image.shape[0]), cv2.INTER_CUBIC
    )

    # save image before adding shadows
    im_cp_one = np.copy(image)
    im_cp_one[mask_out != 0] = 0
    im_out_cp = np.copy(im_out)
    before = im_cp_one.astype("float32") + im_out_cp.astype("float32")

    v_avg = 0.5647  # V value (in HSV) for the green screen, which is #00903a

    # mask image for patch insert
    image_cp = np.copy(image)
    image_cp[mask_out == 0] = 0

    # convert to HSV space for shadow estimation
    target_hsv = cv2.cvtColor(image_cp, cv2.COLOR_BGR2HSV)
    target_hsv = target_hsv.astype("float32")
    target_hsv /= 255.0

    # apply shadows to patch
    ratios = target_hsv[:, :, 2] / v_avg
    im_out = im_out.astype("float32")
    im_out[:, :, 0] = im_out[:, :, 0] * ratios
    im_out[:, :, 1] = im_out[:, :, 1] * ratios
    im_out[:, :, 2] = im_out[:, :, 2] * ratios
    im_out[im_out > 255.0] = 255.0

    im_cp_two = np.copy(image)
    im_cp_two[mask_out != 0] = 0
    final = im_cp_two.astype("float32") + im_out.astype("float32")

    return before, final


def insert_patch(
    gs_coords, gs_im, patch, gs_shape, cc_gt, cc_scene, apply_realistic_effects
):
    """
    :param gs_coords: green screen coordinates in [(x0,y0),(x1,y1),...] format. Type ndarray.
    :param gs_im: clean image with green screen. Type ndarray.
    :param patch: adversarial patch. Type ndarray
    :param gs_shape: green screen shape. Type str.
    :param cc_gt: colorchecker ground truth values. Type ndarray.
    :param cc_scene: colorchecker values in the scene. Type ndarray.
    :param apply_realistic_effects: apply effects such as color correction, blurring, and shadowing. Type bool.
    """

    if apply_realistic_effects:
        # calculate color matrix
        ccm = calculate_ccm(cc_scene, cc_gt, Vandermonde=True)

        # get edge coords
        edge_coords = [
            (gs_coords[0, 0] + 10, gs_coords[0, 1] - 20),
            (gs_coords[0, 0] + 10, gs_coords[0, 1] + 20),
        ]
        edge_coords = np.array(edge_coords)

        # apply color matrix to patch
        patch = apply_ccm(patch.astype("float32"), ccm)

        # resize patch and apply blurring
        scale = (np.amax(gs_coords[:, 1]) - np.amin(gs_coords[:, 1])) / patch.shape[0]
        patch = cv2.resize(
            patch, (int(patch.shape[1] * scale), int(patch.shape[0] * scale))
        )
        patch = find_blur(gs_im, patch, coords=edge_coords)

        # datatype correction
        patch = patch * 255
    patch = patch.astype("uint8")

    # convert for use with cv2
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    # insert transformed patch
    image_digital, image_physical = insert_transformed_patch(
        patch, gs_im, gs_shape, image_coords=gs_coords
    )

    if apply_realistic_effects:
        return image_physical
    else:
        return image_digital


class DApricotPatch(RobustDPatchTargeted):
    """
    """

    def __init__(self, estimator, **kwargs):
        # set batch_size to allow for calculating universal patch over all three cameras
        super().__init__(estimator=estimator, **kwargs)

    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(
            isinstance(s, int) for s in self.patch_shape
        ):
            raise ValueError(
                "The patch shape must be either a tuple or list of integers."
            )
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if not isinstance(self.patch_location, (tuple, list)) or not all(
            isinstance(s, int) for s in self.patch_location
        ):
            raise ValueError(
                "The patch location must be either a tuple or list of integers."
            )
        if len(self.patch_location) != 2:
            raise ValueError("The length of patch location must be 2.")

        if not isinstance(self.crop_range, (tuple, list)) or not all(
            isinstance(s, int) for s in self.crop_range
        ):
            raise ValueError(
                "The crop range must be either a tuple or list of integers."
            )
        if len(self.crop_range) != 2:
            raise ValueError("The length of crop range must be 2.")

        if self.crop_range[0] > self.crop_range[1]:
            raise ValueError(
                "The first element of the crop range must be less or equal to the second one."
            )

        if (
            self.patch_location[0] < self.crop_range[0]
            or self.patch_location[1] < self.crop_range[1]
        ):
            raise ValueError("The patch location must be outside the crop range.")

        if not isinstance(self.brightness_range, (tuple, list)) or not all(
            isinstance(s, float) for s in self.brightness_range
        ):
            raise ValueError(
                "The brightness range must be either a tuple or list of floats."
            )
        if len(self.brightness_range) != 2:
            raise ValueError("The length of brightness range must be 2.")

        # if self.brightness_range[0] < 0.0 or self.brightness_range[1] > 1.0:
        #     raise ValueError("The brightness range must be between 0.0 and 1.0.")

        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError(
                "The first element of the brightness range must be less or equal to the second one."
            )

        if not isinstance(self.rotation_weights, (tuple, list)) or not all(
            isinstance(s, (float, int)) for s in self.rotation_weights
        ):
            raise ValueError(
                "The rotation sampling weights must be provided as tuple or list of float or int values."
            )
        if len(self.rotation_weights) != 4:
            raise ValueError("The number of rotation sampling weights must be 4.")

        if not all(s >= 0.0 for s in self.rotation_weights):
            raise ValueError("The rotation sampling weights must be non-negative.")

        if all(s == 0.0 for s in self.rotation_weights):
            raise ValueError(
                "At least one of the rotation sampling weights must be strictly greater than zero."
            )

        if not isinstance(self.sample_size, int):
            raise ValueError("The EOT sample size must be of type int.")
        if self.sample_size <= 0:
            raise ValueError("The EOT sample size must be greater than 0.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")

    def _augment_images_with_patch(self, x, y, patch, channels_first):
        """
        Augment images with patch using perspective transform
        instead of inserting patch based on patch_location

        :param x: Sample images.
        :param y: Target labels.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        if x.shape[0] != len(self.gs_coords):
            raise ValueError(
                "Number of images should be equal to the number of arrays of green screen coordinates"
            )
        if y is not None and (x.shape[0] != len(y)):
            raise ValueError(
                "Number of images should be equal to the number of targets"
            )

        transformations = dict()
        x_copy = x.copy()
        patch_copy = patch.copy()

        # Apply patch:
        x_patch = []
        for xi, gs_coords in zip(x_copy, self.gs_coords):
            img_with_patch = insert_patch(
                gs_coords,
                xi[:, :, ::-1] * 255.0,  # input image needs to be BGR
                patch_copy * self.mask * 255.0,
                self.patch_geometric_shape,
                cc_gt=None,
                cc_scene=None,
                apply_realistic_effects=False,
            )
            x_patch.append(img_with_patch[:, :, ::-1] / 255.0)  # convert back to RGB
        x_patch = np.asarray(x_patch)

        # 1) crop images: not used.
        if self.crop_range[0] != 0 and self.crop_range[1] != 0:
            logger.warning("crop_range argument not used.")

        # 2) rotate images:
        if sum(self.rotation_weights[1:]) > 0:
            raise ValueError("Non-zero rotations not correctly supported at this time.")

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = brightness * x_copy
        x_patch = brightness * x_patch

        transformations.update({"brightness": brightness})

        patch_target = list()

        for i_image in range(x_copy.shape[0]):
            target_dict = dict()
            target_dict["boxes"] = y[i_image]["boxes"]
            target_dict["labels"] = y[i_image]["labels"]
            target_dict["scores"] = y[i_image]["scores"]

            patch_target.append(target_dict)

        return x_patch, patch_target, transformations

    def _untransform_gradients(
        self, gradients, transforms, channels_first,
    ):
        """
        Revert transformation on gradients using perspective transform

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """
        if gradients.shape[0] != len(self.gs_coords):
            raise ValueError(
                "Number of gradient arrays should be equal to the number of arrays of green screen coordinates"
            )

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo perspective transform for gradients
        patch_coords = shape_coords(
            self.patch_shape[0], self.patch_shape[1], self.patch_geometric_shape
        )
        gradients_tmp = []
        for grads, gs_coords in zip(gradients, self.gs_coords):
            h, _ = cv2.findHomography(gs_coords, patch_coords)
            grads_tmp = cv2.warpPerspective(
                grads, h, (self.mask.shape[1], self.mask.shape[0]), cv2.INTER_CUBIC
            )
            gradients_tmp.append(grads_tmp)
        gradients = np.asarray(gradients_tmp)
        gradients = gradients * self.mask

        return gradients

    def generate(
        self, x, y_object=None, y_patch_metadata=None, threat_model="physical"
    ):
        num_imgs = x.shape[0]
        attacked_images = []

        if threat_model == "digital":
            # Each image in the D-APRICOT 3-tuple of images is attacked individually
            if self.batch_size != 1:
                raise ValueError(
                    'DApricotPatch digital attack requires attack["kwargs"]["batch_size"] == 1'
                )

            for i in range(num_imgs):
                gs_coords = y_patch_metadata[i]["gs_coords"]
                patch_width = np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0])
                patch_height = np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1])
                patch_dim = max(patch_height, patch_width)
                # must be square for now, until it's determined why non-square patches
                # fail to completely overlay green screen
                self.patch_shape = (patch_dim, patch_dim, 3)

                self.patch_geometric_shape = (
                    y_patch_metadata[i]["shape"].tobytes().decode("utf-8")
                )

                self.mask = create_mask(
                    self.patch_geometric_shape, self.patch_shape[0], self.patch_shape[1]
                )

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
                self._patch = self._patch * self.mask

                self.gs_coords = [gs_coords]

                patch = super().generate(np.expand_dims(x[i], axis=0), y=[y_object[i]])

                # apply patch using perspective transform only
                img_with_patch = insert_patch(
                    self.gs_coords[0],
                    x[i][:, :, ::-1] * 255.0,  # image needs to be BGR
                    patch * self.mask * 255.0,
                    self.patch_geometric_shape,
                    cc_gt=None,
                    cc_scene=None,
                    apply_realistic_effects=False,
                )
                attacked_images.append(img_with_patch[:, :, ::-1] / 255.0)

        else:
            if self.batch_size != 3:
                raise ValueError(
                    'DApricotPatch physical attack requires attack["kwargs"]["batch_size"] == 3'
                )

            # generate universal patch for all three cameras
            gs_coords = [y_patch_metadata[i]["gs_coords"] for i in range(num_imgs)]

            patch_width = max(
                [
                    np.max(gs_coords[i][:, 0]) - np.min(gs_coords[i][:, 0])
                    for i in range(num_imgs)
                ]
            )
            patch_height = max(
                [
                    np.max(gs_coords[i][:, 1]) - np.min(gs_coords[i][:, 1])
                    for i in range(num_imgs)
                ]
            )
            patch_dim = max(patch_height, patch_width)
            # must be square for now, until it's determined why non-square patches
            # fail to completely overlay green screen
            self.patch_shape = (patch_dim, patch_dim, 3)

            self.patch_geometric_shape = (
                y_patch_metadata[0]["shape"].tobytes().decode("utf-8")
            )

            self.mask = create_mask(
                self.patch_geometric_shape, self.patch_shape[0], self.patch_shape[1]
            )

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
            self._patch = self._patch * self.mask

            self.gs_coords = gs_coords

            patch = super().generate(x, y=[y_object[i] for i in range(num_imgs)])

            # apply universal patch to all images
            for i in range(num_imgs):
                cc_gt = y_patch_metadata[i]["cc_ground_truth"]
                cc_scene = y_patch_metadata[i]["cc_scene"]
                img_with_patch = insert_patch(
                    self.gs_coords[i],
                    x[i][:, :, ::-1] * 255.0,  # input must be BGR
                    patch * self.mask * 255.0,
                    self.patch_geometric_shape,
                    cc_gt=cc_gt,
                    cc_scene=cc_scene,
                    apply_realistic_effects=True,
                )
                attacked_images.append(img_with_patch[:, :, ::-1] / 255.0)
        return np.array(attacked_images)


class DApricotMaskedPGD(ProjectedGradientDescent):
    """
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

    def generate(self, x, y_object=None, y_patch_metadata=None, threat_model="digital"):
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
                    np.expand_dims(x[i], axis=0), y=[y_object[i]], mask=img_mask
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
        h, _ = cv2.findHomography(patch_coords, gs_coords)
        inscribed_patch_mask = create_mask(shape, gs_size, gs_size)
        img_mask = cv2.warpPerspective(
            inscribed_patch_mask, h, (x.shape[1], x.shape[0]), cv2.INTER_CUBIC
        )
        return img_mask
