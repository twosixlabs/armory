"""
This file is used to compute the color transform for more realistic
digital patch insertion.

Contributed by The MITRE Corporation, 2021
"""

import numpy.core.multiarray
import os
import csv
import cv2
import json
import math
import colour
import random
import string
import pandas
import argparse
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


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
    smallest_dim = min(h, w)
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
    if coords == []:
        # get points for finding blur
        circle_image = np.copy(image)
        points = Points(circle_image)
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", points.draw_circle)

        while True:
            cv2.imshow("Image", circle_image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        coords = np.array(points.points)

    # convert image to grayscale and normalize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_chip = gray[coords[0, 1] : coords[1, 1], coords[0, 0]].astype("float32")
    gray_norm = gray_chip / 255.0

    # differentiate to estimate lsf
    lsf = np.diff(gray_norm)
    lsf[lsf < 0.0] = 0.0
    norm_lsf = lsf / np.max(lsf)

    # fit gaussian to lsf data
    mean, std = norm.fit(norm_lsf)

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
    breakpoint()
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

    # calculate color matrix
    ccm = calculate_ccm(cc_scene, cc_gt, Vandermonde=True)

    # get edge coords
    edge_coords = [
        (gs_coords[0, 0] + 10, gs_coords[0, 1] - 20),
        (gs_coords[0, 0] + 10, gs_coords[0, 1] + 20),
    ]
    edge_coords = np.array(edge_coords)

    if apply_realistic_effects:
        # apply color matrix to patch
        patch = apply_ccm(patch.astype("float32"), ccm)

    if apply_realistic_effects:
        # resize patch and apply blurring
        scale = (np.amax(gs_coords[:, 1]) - np.amin(gs_coords[:, 1])) / patch.shape[0]
        patch = cv2.resize(
            patch, (int(patch.shape[1] * scale), int(patch.shape[0] * scale))
        )
        patch = find_blur(gs_im, patch, coords=edge_coords)

        # datatype correction
    if apply_realistic_effects:
        patch = patch * 255
    patch = patch.astype("uint8")

    # convert for use with cv2
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    # insert transformed patch
    image_digital, image_physical = insert_transformed_patch(
        patch, gs_im, gs_shape, image_coords=gs_coords
    )

    if apply_realistic_effects:
        return image_physical[:, :, ::-1]
    else:
        return image_digital[:, :, ::-1]
