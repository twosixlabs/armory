import numpy as np


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
