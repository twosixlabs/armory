import math
import os
import random
from typing import Dict, List, Optional

from art.attacks.evasion import RobustDPatch
import cv2
import numpy as np
from tqdm.auto import trange

from armory.art_experimental.attacks.carla_obj_det_utils import (
    linear_depth_to_rgb,
    linear_to_log,
    log_to_linear,
    rgb_depth_to_linear,
)
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


class CARLADapricotPatch(RobustDPatch):
    def __init__(self, estimator, **kwargs):
        # Maximum depth perturbation from a flat patch
        self.depth_delta_meters = kwargs.pop("depth_delta_meters", 3)
        self.learning_rate_depth = kwargs.pop("learning_rate_depth", 0.0001)
        self.depth_perturbation = None
        self.max_depth = None
        self.min_depth = None
        self.patch_base_image = kwargs.pop("patch_base_image", None)

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

        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        # find the colors within the boundaries
        mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
        mask = np.expand_dims(mask, 2)
        # cv2.imwrite(
        #     "mask.png", mask
        # )  # visualize perturbable regions. Comment out if not needed.

        patch_base = im / 255.0
        mask = mask / 255.0
        return patch_base, mask

    def inner_generate(  # type: ignore
        self, x: np.ndarray, y: Optional[List[Dict[str, np.ndarray]]] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate RobustDPatch.
        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
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
            patch_gradients = np.zeros_like(self._patch)
            depth_gradients = None

            for e_step in range(self.sample_size):

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    if y is None:
                        y_batch = y
                    else:
                        y_batch = y[i_batch_start:i_batch_end]

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
                    )  # (B,H,W,C)

                    gradients_rgb = gradients[:, :, :, :3]
                    if gradients.shape[-1] == 6:
                        if depth_gradients is None:
                            depth_gradients = gradients[:, :, :, 3:]
                        else:
                            depth_gradients = np.concatenate(
                                (depth_gradients, gradients[:, :, :, 3:]), axis=0
                            )

                    gradients_rgb = self._untransform_gradients(
                        gradients_rgb,
                        transforms,
                        channels_first=self.estimator.channels_first,
                    )
                    patch_gradients = patch_gradients + np.sum(gradients_rgb, axis=0)

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
                * self.patch_mask
            )

            # Update depth perturbation
            if gradients.shape[-1] == 6:
                images_depth = patched_images[:, :, :, 3:]

                if self.depth_type == "log":
                    depth_log = (
                        self.depth_perturbation
                        + np.sign(depth_gradients)
                        * (1 - 2 * int(self.targeted))
                        * self.learning_rate_depth
                    )
                    perturbed_images = np.clip(
                        images_depth + depth_log, self.min_depth, self.max_depth
                    )
                    self.depth_pertubation = perturbed_images - images_depth
                else:
                    grads_linear = rgb_depth_to_linear(
                        depth_gradients[:, :, :, 0],
                        depth_gradients[:, :, :, 1],
                        depth_gradients[:, :, :, 2],
                    ).astype("float32")
                    depth_linear = rgb_depth_to_linear(
                        self.depth_perturbation[:, :, :, 0],
                        self.depth_perturbation[:, :, :, 1],
                        self.depth_perturbation[:, :, :, 2],
                    ).astype("float32")
                    depth_linear = (
                        depth_linear
                        + np.sign(grads_linear)
                        * (1 - 2 * int(self.targeted))
                        * self.learning_rate_depth
                    )

                    images_depth_linear = rgb_depth_to_linear(
                        images_depth[:, :, :, 0],
                        images_depth[:, :, :, 1],
                        images_depth[:, :, :, 2],
                    ).astype("float32")
                    depth_linear = np.clip(
                        images_depth_linear + depth_linear,
                        self.min_depth,
                        self.max_depth,
                    )
                    depth_r, depth_g, depth_b = linear_depth_to_rgb(depth_linear)
                    perturbed_images = np.stack(
                        [depth_r, depth_g, depth_b], axis=-1
                    ).astype("float32")
                    self.depth_perturbation = perturbed_images - images_depth

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
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
        for i in range(len(x_copy)):
            xi = x_copy[i]
            gs_coords = self.gs_coords[i]

            if xi.shape[-1] == 3:
                rgb_img = xi.astype("float32")
            else:
                rgb_img = xi[:, :, :3].astype("float32")  # (H,W,3)
                depth_img = xi[:, :, 3:].astype("float32")  # (H,W,3)
                depth_perturbation = self.depth_perturbation[i]  # (H,W,3)
                foreground = self.foreground[i]  # (H,W,1)

            # apply patch using DAPRICOT transform to RGB channels only
            # insert_patch() uses BGR color ordering for input and output
            rgb_img_with_patch = (
                insert_patch(
                    gs_coords,
                    rgb_img[:, :, ::-1] * 255.0,  # input image needs to be BGR
                    patch_copy * 255.0,
                    self.patch_geometric_shape,
                    cc_gt=self.cc_gt,
                    cc_scene=self.cc_scene,
                    apply_realistic_effects=self.apply_realistic_effects,
                    rgb=True,
                )
                / 255.0
            )

            # embed patch into background
            rgb_img_with_patch[
                np.all(self.binarized_patch_mask == 0, axis=-1)
            ] = rgb_img[np.all(self.binarized_patch_mask == 0, axis=-1)][
                :, ::-1
            ]  # (H,W,3)

            if xi.shape[-1] == 3:
                img_with_patch = rgb_img_with_patch.copy()
            else:
                # apply depth perturbation to depth channels
                depth_img = depth_img + depth_perturbation * foreground  # (H,W,3)

                depth_img = np.clip(
                    depth_img,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                )

                img_with_patch = np.concatenate(
                    (depth_img[:, :, ::-1], rgb_img_with_patch), axis=-1
                )

            x_patch.append(img_with_patch[:, :, ::-1])  # convert back to RGB
        x_patch = np.asarray(x_patch)  # (B,H,W,3)

        # 1) crop images: not used.
        if self.crop_range[0] != 0 and self.crop_range[1] != 0:
            log.warning("crop_range argument not used.")

        # 2) rotate images
        if sum(self.rotation_weights[1:]) > 0:
            raise ValueError("Non-zero rotations not correctly supported at this time.")

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy[..., :3] = brightness * x_copy[..., :3]
        x_patch[..., :3] = brightness * x_patch[..., :3]

        transformations.update({"brightness": brightness})

        patch_target = list()

        if not self.targeted:
            y = self.estimator.predict(
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
        if gradients.shape[0] != len(self.gs_coords):
            raise ValueError(
                "Number of gradient arrays should be equal to the number of arrays of green screen coordinates"
            )

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo perspective transform for gradients
        h, w, c = self.patch_shape
        patch_coords = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        gradients_tmp = []
        for grads, gs_coords in zip(gradients, self.gs_coords):
            h, _ = cv2.findHomography(gs_coords, patch_coords)
            grads_tmp = cv2.warpPerspective(
                grads, h, (self.patch_shape[1], self.patch_shape[0]), cv2.INTER_CUBIC
            )
            gradients_tmp.append(grads_tmp)
        gradients = np.asarray(gradients_tmp)

        return gradients

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
        if x.shape[-1] == 6 and self.sample_size != 1:
            raise ValueError(
                "Sample size must be 1 for multimodality input because expectation over transformation cannot be applied to depth perturbation"
            )

        num_imgs = x.shape[0]
        attacked_images = []

        for i in range(num_imgs):

            if x.shape[-1] == 3:
                rgb_img = x[i].astype("float32")
            else:
                rgb_img = x[i, :, :, :3].astype("float32")  # (H,W,3)
                depth_img = x[i, :, :, 3:].astype("float32")  # (H,W,3)

            gs_coords = y_patch_metadata[i]["gs_coords"]  # (4,2)
            patch_width = np.max(gs_coords[:, 0]) - np.min(gs_coords[:, 0])
            patch_height = np.max(gs_coords[:, 1]) - np.min(gs_coords[:, 1])
            self.patch_shape = (
                patch_height,
                patch_width,
                3,
            )

            patch_geometric_shape = y_patch_metadata[i].get("shape", "rect")
            self.patch_geometric_shape = str(patch_geometric_shape)

            # this masked to embed patch into the background in the event of occlusion
            self.binarized_patch_mask = y_patch_metadata[i]["mask"]  # (H,W,3)

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
                self._patch, self.patch_mask = self.create_initial_image(
                    (patch_width, patch_height),
                    self.hsv_lower_bound,
                    self.hsv_upper_bound,
                )
            else:
                self._patch = (
                    np.random.randint(0, 255, size=self.patch_shape)
                    / 255
                    * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                    + self.estimator.clip_values[0]
                )
                self.patch_mask = np.ones_like(self._patch)

            self._patch = np.clip(
                self._patch,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )

            self.gs_coords = [gs_coords]

            # initialize depth variables
            if x.shape[-1] == 6:

                if np.all(x[i, :, :, 3] == x[i, :, :, 4]) and np.all(
                    x[i, :, :, 3] == x[i, :, :, 5]
                ):
                    self.depth_type = "log"
                    depth_linear = log_to_linear(x[i, :, :, 3:])
                    self.max_depth = np.minimum(
                        1.0, linear_to_log(depth_linear + self.depth_delta_meters)
                    )
                    self.min_depth = np.maximum(
                        0.0, linear_to_log(depth_linear - self.depth_delta_meters)
                    )
                else:
                    self.depth_type = "linear"
                    depth_linear = rgb_depth_to_linear(
                        x[i, :, :, 3], x[i, :, :, 4], x[i, :, :, 5]
                    )
                    self.max_depth = np.minimum(
                        1000.0, depth_linear + self.depth_delta_meters
                    )
                    self.min_depth = np.maximum(
                        0.0, depth_linear - self.depth_delta_meters
                    )

                self.depth_perturbation = np.zeros(
                    (1, *x.shape[1:3], 3), dtype=np.float32
                )  # (1,H,W,3)
                self.foreground = np.all(self.binarized_patch_mask == 255, axis=-1)
                self.foreground = np.expand_dims(self.foreground, (-1, 0))  # (1,H,W,1)
                # ensure area perturbed in depth is consistent with area perturbed in RGB
                h, _ = cv2.findHomography(
                    np.array(
                        [
                            [0, 0],
                            [patch_width - 1, 0],
                            [patch_width - 1, patch_height - 1],
                            [0, patch_height - 1],
                        ]
                    ),
                    gs_coords,
                )
                rgb_mask = np.ones((patch_height, patch_width, 3), dtype=np.float32)
                rgb_mask = cv2.warpPerspective(
                    rgb_mask, h, (x.shape[2], x.shape[1]), cv2.INTER_CUBIC
                )
                self.foreground = self.foreground * rgb_mask[:, :, 0:1]

            if y is None:
                patch = self.inner_generate(
                    np.expand_dims(x[i], axis=0)
                )  # untargeted attack
            else:
                patch = self.inner_generate(
                    np.expand_dims(x[i], axis=0), y=[y[i]]
                )  # targeted attack

            rgb_img_with_patch = (
                insert_patch(
                    self.gs_coords[0],
                    rgb_img[:, :, ::-1] * 255.0,
                    patch * 255.0,
                    self.patch_geometric_shape,
                    cc_gt=self.cc_gt,
                    cc_scene=self.cc_scene,
                    apply_realistic_effects=self.apply_realistic_effects,
                    rgb=True,
                )
                / 255.0
            )

            # embed patch into background
            rgb_img_with_patch[
                np.all(self.binarized_patch_mask == 0, axis=-1)
            ] = rgb_img[np.all(self.binarized_patch_mask == 0, axis=-1)][
                :, ::-1
            ]  # (H,W,3)

            if x.shape[-1] == 3:
                img_with_patch = rgb_img_with_patch.copy()
            else:
                depth_img = (
                    depth_img + self.depth_perturbation[0] * self.foreground[0]
                )  # (H,W,3)

                depth_img = np.clip(
                    depth_img,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                )

                img_with_patch = np.concatenate(
                    (depth_img[:, :, ::-1], rgb_img_with_patch), axis=-1
                )
            attacked_images.append(img_with_patch[:, :, ::-1])

        return np.array(attacked_images)  # (B,H,W,3)
