"""
This file is used to compute the color transform for more realistic
digital patch insertion.

Contributed by The MITRE Corporation, 2021
"""

import cv2
import math
import numpy as np


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


def insert_patch(gs_coords, gs_im, patch, gs_shape):
    """
    :param gs_coords: green screen coordinates in [(x0,y0),(x1,y1),...] format. Type ndarray.
    :param gs_im: clean image with green screen. Type ndarray.
    :param patch: adversarial patch. Type ndarray
    :param gs_shape: green screen shape. Type str.
    :param cc_gt: colorchecker ground truth values. Type ndarray.
    :param cc_scene: colorchecker values in the scene. Type ndarray.
    :param apply_realistic_effects: apply effects such as color correction, blurring, and shadowing. Type bool.
    """
    patch = patch.astype("uint8")

    # convert for use with cv2
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    # insert transformed patch
    image_digital, image_physical = insert_transformed_patch(
        patch, gs_im, gs_shape, image_coords=gs_coords
    )
    return image_digital
