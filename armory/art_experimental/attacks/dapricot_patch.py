"""
Copyright 2021 The MITRE Corporation. All rights reserved
"""
import math

from art.attacks.evasion import ProjectedGradientDescent
import cv2
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


class DApricotMaskedPGD(ProjectedGradientDescent):
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
