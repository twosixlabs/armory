import math
import os
import random
from typing import Dict, List, Optional

from art.attacks.evasion import RobustDPatch
import cv2
import numpy as np
from tqdm.auto import trange

from armory.logs import log
from armory.utils.external_repo import ExternalRepoImport

with ExternalRepoImport(
    repo="colour-science/colour@v0.3.16",
    experiment="carla_obj_det_dpatch_undefended.json",
):
    import colour


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
        ccm = colour.characterisation.matrix_colour_correction_Vandermonde(
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


def insert_transformed_patch(
    patch, image, gs_shape, rgb, patch_coords=[], image_coords=[]
):
    """
    Insert patch to image based on given or selected coordinates

    attributes::
        patch
            patch as numpy array
        image
            image as numpy array
        gs_shape
            green screen shape
        rgb
            true if gs_im is RGB, false otherwise
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
        h, w, c = patch.shape
        patch_coords = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # calculate homography
    h, status = cv2.findHomography(patch_coords, image_coords)

    # mask to aid with insertion
    mask = np.ones((patch.shape[0], patch.shape[1], 3))
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

    if rgb:
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
    else:
        final = before

    return before, final


def insert_patch(
    gs_coords, gs_im, patch, gs_shape, cc_gt, cc_scene, apply_realistic_effects, rgb
):
    """
    :param gs_coords: green screen coordinates in [(x0,y0),(x1,y1),...] format. Type ndarray.
    :param gs_im: clean image with green screen. Type ndarray.
    :param patch: adversarial patch. Type ndarray
    :param gs_shape: green screen shape. Type str.
    :param cc_gt: colorchecker ground truth values. Type ndarray.
    :param cc_scene: colorchecker values in the scene. Type ndarray.
    :param apply_realistic_effects: apply effects such as color correction, blurring, and shadowing. Type bool.
    :param rgb: true if gs_im is RGB, false otherwise
    """

    if apply_realistic_effects:
        # calculate color matrix
        ccm = calculate_ccm(cc_scene, cc_gt, Vandermonde=True)

        # apply color matrix to patch
        patch = apply_ccm(patch.astype("float32"), ccm)

        # resize patch
        scale = (np.amax(gs_coords[:, 1]) - np.amin(gs_coords[:, 1])) / patch.shape[0]
        patch = cv2.resize(
            patch,
            (int(patch.shape[1] * scale), int(patch.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

        # datatype correction
        patch = patch * 255
    patch = patch.astype("uint8")

    # convert for use with cv2
    if rgb:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    enlarged_coords = np.copy(gs_coords)
    pad_amt_x = int(0.03 * (enlarged_coords[2, 0] - enlarged_coords[0, 0]))
    pad_amt_y = int(0.03 * (gs_coords[2, 1] - gs_coords[0, 1]))
    enlarged_coords[0, 0] -= pad_amt_x
    enlarged_coords[0, 1] -= pad_amt_y
    enlarged_coords[1, 0] += pad_amt_x
    enlarged_coords[1, 1] -= pad_amt_y
    enlarged_coords[2, 0] += pad_amt_x
    enlarged_coords[2, 1] += pad_amt_y
    enlarged_coords[3, 0] -= pad_amt_x
    enlarged_coords[3, 1] += pad_amt_y

    # insert transformed patch
    image_digital, image_physical = insert_transformed_patch(
        patch, gs_im, gs_shape, image_coords=enlarged_coords, rgb=rgb
    )

    if apply_realistic_effects:
        return image_physical
    else:
        return image_digital


class CARLAMOTDapricotPatch(RobustDPatch):
    def __init__(self, estimator, **kwargs):
        # initialize batch_size with batch_frame_size so it's clear to users
        # that frames, not videos, are batched
        kwargs["batch_size"] = kwargs.pop("batch_frame_size", 1)
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

        patch_base = im / 255.0
        return patch_base

    def inner_generate(  # type: ignore
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate RobustDPatch.
        :param x: Sample images with shape (NHWC)
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """

        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
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
        if x.ndim != 4:  # pragma: no cover
            raise ValueError("The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images:
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        if not self.estimator.native_label_is_pytorch_format and y is not None:
            from art.estimators.object_detection.utils import convert_tf_to_pt

            y = convert_tf_to_pt(y=y, height=x.shape[1], width=x.shape[2])

        if y is not None:
            for i_image in range(x.shape[0]):
                y_i = y[i_image]["boxes"]
                for i_box in range(y_i.shape[0]):
                    x_1, y_1, x_2, y_2 = y_i[i_box]
                    if (  # pragma: no cover
                        x_1 < self.crop_range[1]
                        or y_1 < self.crop_range[0]
                        or x_2 > image_width - self.crop_range[1] + 1
                        or y_2 > image_height - self.crop_range[0] + 1
                    ):
                        raise ValueError(
                            "Cropping is intersecting with at least one box, reduce `crop_range`."
                        )

        if (  # pragma: no cover
            self.patch_location[0] + self.patch_shape[0]
            > image_height - self.crop_range[0]
            or self.patch_location[1] + self.patch_shape[1]
            > image_width - self.crop_range[1]
        ):
            raise ValueError("The patch (partially) lies outside the cropped image.")

        for i_step in trange(
            self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose
        ):
            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)

            for e_step in range(self.sample_size):

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    if y is None:
                        y_batch = y
                    else:
                        y_batch = y[i_batch_start:i_batch_end]

                    self.patch_coords_batch = self.gs_coords[
                        i_batch_start:i_batch_end, :, :
                    ]
                    self.patch_masks_batch = self.patch_masks_video[
                        i_batch_start:i_batch_end, :, :, :
                    ]

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
                        x=patched_images,
                        y=patch_target,
                        standardise_output=True,
                    )

                    gradients = self._untransform_gradients(
                        gradients,
                        transforms,
                        channels_first=self.estimator.channels_first,
                    )

                    patch_gradients = patch_gradients_old + np.sum(gradients, axis=0)

                    patch_gradients_old = patch_gradients

            # Write summary
            if self.summary_writer is not None:  # pragma: no cover
                x_patched, y_patched, _ = self._augment_images_with_patch(
                    x, y, self._patch, channels_first=self.estimator.channels_first
                )

                self.summary_writer.update(
                    batch_id=0,
                    global_step=i_step,
                    grad=np.expand_dims(patch_gradients, axis=0),
                    patch=self._patch,
                    estimator=self.estimator,
                    x=x_patched,
                    y=y_patched,
                    targeted=self.targeted,
                )

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

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return self._patch

    def _augment_images_with_patch(self, x, y, patch, channels_first):
        """
        Augment images with patch using perspective transform
        instead of inserting patch based on patch_location

        :param x: Sample images.
        :param y: Target labels.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        if x.shape[0] != len(self.patch_coords_batch):
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
        for xi, gs_coords, patch_mask in zip(
            x_copy, self.patch_coords_batch, self.patch_masks_batch
        ):

            rgb_img = (xi * 255.0).astype("float32")

            # apply patch using DAPRICOT transform to RGB channels only
            # insert_patch() uses BGR color ordering for input and output
            rgb_img_with_patch = insert_patch(
                gs_coords,
                rgb_img[:, :, ::-1],  # input image needs to be BGR
                patch_copy * 255.0,
                self.patch_geometric_shape,
                cc_gt=self.cc_gt,
                cc_scene=self.cc_scene,
                apply_realistic_effects=self.apply_realistic_effects,
                rgb=True,
            )

            # embed patch into background
            rgb_img_with_patch[np.all(patch_mask == 0, axis=-1)] = rgb_img[
                np.all(patch_mask == 0, axis=-1)
            ][:, ::-1]

            img_with_patch = rgb_img_with_patch.copy()

            x_patch.append(img_with_patch[:, :, ::-1] / 255.0)  # convert back to RGB
        x_patch = np.asarray(x_patch)

        # 1) crop images: not used.
        if self.crop_range[0] != 0 and self.crop_range[1] != 0:
            log.warning("crop_range argument not used.")

        # 2) rotate images
        if sum(self.rotation_weights[1:]) > 0:
            raise ValueError("Non-zero rotations not correctly supported at this time.")

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = brightness * x_copy
        x_patch = brightness * x_patch

        transformations.update({"brightness": brightness})

        patch_target = list()

        if not self.targeted:
            y = self.estimator.predict_object_detector(
                x=x_copy.astype("float32"), standardise_output=True
            )

        for i_image in range(x_copy.shape[0]):
            target_dict = dict()
            target_dict["boxes"] = y[i_image]["boxes"]
            target_dict["labels"] = y[i_image]["labels"]
            target_dict["scores"] = y[i_image]["scores"]

            patch_target.append(target_dict)

        return x_patch, patch_target, transformations

    def _untransform_gradients(
        self,
        gradients,
        transforms,
        channels_first,
    ):
        """
        Revert transformation on gradients using perspective transform

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """
        if gradients.shape[0] != len(self.patch_coords_batch):
            raise ValueError(
                "Number of gradient arrays should be equal to the number of arrays of green screen coordinates"
            )

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo perspective transform for gradients
        h, w, c = self.patch_shape
        patch_coords = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        gradients_tmp = []
        for grads, gs_coords in zip(gradients, self.patch_coords_batch):
            h, _ = cv2.findHomography(gs_coords, patch_coords)
            grads_tmp = cv2.warpPerspective(
                grads, h, (self.patch_shape[1], self.patch_shape[0]), cv2.INTER_CUBIC
            )
            gradients_tmp.append(grads_tmp)
        gradients = np.asarray(gradients_tmp)

        return gradients

    def generate(self, x, y=None, y_patch_metadata=None):
        """
        param x: Sample video of shape=(NFHWC).
        param y: [Optional] Sample labels. List of dictionaries,
            ith dictionary contains bounding boxes, class labels, and class scores
        param y_patch_metadata: Patch metadata. List of N dictionaries, ith dictionary contains patch metadata for x[i]
        """
        if x.shape[0] > 1:
            log.info("To perform per-example patch attack, batch size must be 1")
        assert x.shape[-1] == 3, "x must have 3 color channels"

        num_vids = x.shape[0]
        attacked_images = []

        for i in range(num_vids):
            rgb_imgs = (x[i] * 255.0).astype("float32")
            self.gs_coords = y_patch_metadata[i]["gs_coords"]
            patch_width = np.max(self.gs_coords[:, :, 0]) - np.min(
                self.gs_coords[:, :, 0]
            )
            patch_height = np.max(self.gs_coords[:, :, 1]) - np.min(
                self.gs_coords[:, :, 1]
            )
            self.patch_shape = (
                patch_height,
                patch_width,
                x.shape[-1],
            )

            patch_geometric_shape = y_patch_metadata[i].get("shape", "rect")
            self.patch_geometric_shape = str(patch_geometric_shape)

            # this masked to embed patch into the background in the event of occlusion
            self.patch_masks_video = y_patch_metadata[i]["masks"]

            # get colorchecker information from ground truth and scene
            self.cc_gt = y_patch_metadata[i].get("cc_ground_truth", None)
            self.cc_scene = y_patch_metadata[i].get("cc_scene", None)

            # Prior to Eval 5 (dev set version 2.0.0), this was set to True. It's now being
            # set False to be more aligned with AdversarialPatch attack. This line is essentially
            # 'if dev set version >= 2.0.0', since these variables are None as of 2.0.0 / Eval 5
            if self.cc_gt is None or self.cc_scene is None:
                self.apply_realistic_effects = False
            else:
                self.apply_realistic_effects = True

            # self._patch needs to be re-initialized with the correct shape
            if self.patch_base_image is not None:
                self.patch_base = self.create_initial_image(
                    (patch_width, patch_height),
                )
                self._patch = self.patch_base
            else:
                self._patch = (
                    np.random.randint(0, 255, size=self.patch_shape)
                    / 255
                    * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                    + self.estimator.clip_values[0]
                )

            self._patch = np.clip(
                self._patch,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            if y is None:
                patch = self.inner_generate(x[i])  # untargeted attack
            else:
                patch = self.inner_generate(x[i], y=y[i])  # targeted attack

            # apply patch using DAPRICOT transform to RGB channels only
            # insert_patch() uses BGR color ordering for input and output
            rgb_imgs_with_patch = []
            for rgb_img, gs_coords in zip(rgb_imgs, self.gs_coords):
                rgb_img_with_patch = insert_patch(
                    gs_coords,
                    rgb_img[:, :, ::-1],
                    patch * 255.0,
                    self.patch_geometric_shape,
                    cc_gt=self.cc_gt,
                    cc_scene=self.cc_scene,
                    apply_realistic_effects=self.apply_realistic_effects,
                    rgb=True,
                )
                rgb_imgs_with_patch.append(rgb_img_with_patch)
            rgb_imgs_with_patch = np.array(rgb_imgs_with_patch)

            # embed patch into background
            rgb_imgs_with_patch[
                np.all(self.patch_masks_video == 0, axis=-1)
            ] = rgb_imgs[np.all(self.patch_masks_video == 0, axis=-1)][:, ::-1]

            img_with_patch = rgb_imgs_with_patch.copy()

            attacked_images.append(img_with_patch[:, :, :, ::-1] / 255.0)

        return np.array(attacked_images)
