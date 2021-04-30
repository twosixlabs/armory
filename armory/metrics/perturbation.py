"""
Perturbation metrics
"""

import functools
import logging

import numpy as np


logger = logging.getLogger(__name__)


class MetricNameSpace:
    def __setattr__(self, name, function):
        if hasattr(self, name):
            raise ValueError(f"Cannot overwrite existing function {name}")
        if not callable(function):
            raise ValueError(f"{name} function {function} is not callable")
        super().__setattr__(name, function)

    def __delattr__(self, name):
        raise ValueError("Deletion not allowed")

    def _names(self):
        return sorted(x for x in self.__dict__ if not x.startswith("_"))

    def __contains__(self, name):
        return name in self._names()

    def __repr__(self):
        """
        Show the existing non-underscore names
        """
        return str(self._names())


element = MetricNameSpace()
batch = MetricNameSpace()


def as_batch(element_metric):
    """
    Return a batchwise metric function from an elementwise metric function
    """

    @functools.wraps(element_metric)
    def wrapper(x_batch, x_adv_batch, **kwargs):
        x_batch = list(x_batch)
        x_adv_batch = list(x_adv_batch)
        if len(x_batch) != len(x_adv_batch):
            raise ValueError(
                f"len(x_batch) {len(x_batch)} != len(x_adv_batch) {len(x_adv_batch)}"
            )
        y = []
        for x, x_adv in zip(x_batch, x_adv_batch):
            y.append(element_metric(x, x_adv, **kwargs))
        try:
            y = np.array(y)
        except ValueError:
            # Handle ragged arrays
            y = np.array(y, dtype=object)
        return y

    if wrapper.__doc__ is None:
        logger.warning(f"{element_metric.__name__} has no doc string")
        wrapper.__doc__ = ""
    wrapper.__doc__ = "Batch version of:\n" + wrapper.__doc__
    wrapper.__name__ = "batch_" + wrapper.__name__
    # TODO: fix repr(wrapper), which defaults to the element_metric, not __name__
    return wrapper


def batchwise(batch_metric, name=None):
    """
    Register a batch metric and register a batchwise version of it
    """
    if name is None:
        name = batch_metric.__name__
    setattr(batch, name, batch_metric)


def elementwise(element_metric, name=None):
    """
    Register a element metric and register a batchwise version of it
    """
    if name is None:
        name = element_metric.__name__
    setattr(element, name, element_metric)
    batch_metric = as_batch(element_metric)
    batchwise(batch_metric, name=name)
    return element_metric


def numpy(function):
    """
    Ensures args (but not kwargs) are passed in as numpy vectors
    It casts them to float (unless they are complex)
    And it ensures that they have the same shape and dtype
    """

    @functools.wraps(function)
    def wrapper(x, x_adv, **kwargs):
        x, x_adv = (np.asarray(i) for i in (x, x_adv))
        if x.dtype != x_adv.dtype:
            raise ValueError(f"x.dtype {x.dtype} != x_adv.dtype {x_adv.dtype}")
        if x.shape != x_adv.shape:
            raise ValueError(f"x.shape {x.shape} != x_adv.shape {x_adv.shape}")
        if x.dtype == np.complex:  # TODO: Add tests and remove these two lines
            logger.warning("np.complex types are not fully tested.")
        if x.dtype not in (float, np.complex):
            x = x.astype(float)
            x_adv = x_adv.astype(float)
        return function(x, x_adv, **kwargs)

    return wrapper


@numpy
def lp(x, x_adv, *, p=2):
    """
    Return the Lp (vector) norm distance between the input arrays
        Results follow np.linalg.norm

    p - any value castable to float
    """
    p = float(p)
    return np.linalg.norm((x - x_adv).flatten(), ord=p)


@elementwise
def linf(x, x_adv):
    """
    Return the L-infinity norm distance between the input arrays
    """
    return lp(x, x_adv, p=np.inf)


@elementwise
def l2(x, x_adv):
    """
    Return the L2 norm distance between the input arrays
    """
    return lp(x, x_adv, p=2)


@elementwise
def l1(x, x_adv):
    """
    Return the L1 norm distance between the input arrays
    """
    return lp(x, x_adv, p=1)


@elementwise
def l0(x, x_adv):
    """
    Return the L0 'norm' over a batch of inputs as a float,
    normalized by the number of elements in the array
    """
    return lp(x, x_adv, p=0)


@elementwise
@numpy
def snr(x, x_adv):
    """
    Return the absolute SNR of x to (x - x_adv), with range [0, inf]
    """
    signal_power = (np.abs(x) ** 2).mean()
    noise_power = (np.abs(x - x_adv) ** 2).mean()
    if noise_power == 0:
        return np.inf
    else:
        return signal_power / noise_power


@elementwise
@numpy
def snr_spectrogram(x, x_adv):
    """
    Return the SNR of a batch of samples with spectrogram input

    NOTE: Due to phase effects, this is a lower bound of the SNR.
        For instance, if x[0] = sin(t) and x_adv[0] = sin(t + 2*pi/3),
        Then the SNR will be calculated as infinity, when it should be 1.
        However, the spectrograms will look identical, so as long as the
        model uses spectrograms and not the underlying raw signal,
        this should not have impact the results.
    """
    signal_power = np.abs(x).mean()
    noise_power = np.abs(x - x_adv).mean()
    return signal_power / noise_power


def _dB(value):
    """
    Return the input value in dB
    """
    if value < 0:
        raise ValueError(f"dB input must be in range [0, inf], not {value}")
    elif value == 0:
        return -np.inf
    else:
        return 10 * np.log10(value)


@elementwise
def snr_db(x, x_adv):
    """
    Return the absolute SNR of x to (x - x_adv) in decibels (dB)
    """
    return _dB(snr(x, x_adv))


@elementwise
def snr_spectrogram_db(x, x_adv):
    """
    Return the SNR of a batch of samples with spectrogram input in Decibels (DB)
    """
    return _dB(snr_spectrogram(x, x_adv))


def image_rectangular_patch_area(x, x_adv):
    raise NotImplementedError("not completed")


@elementwise
@numpy
def image_circle_patch_diameter(x, x_adv):
    """
    Return the diameter of the smallest circular patch over the modified pixels
    """
    img_shape = x.shape
    if len(img_shape) != 3:
        raise ValueError(f"Expected image with 3 dimensions. x has shape {x.shape}")
    if (x == x_adv).mean() < 0.5:
        logger.warning(
            f"x and x_adv differ at {int(100*(x != x_adv).mean())} percent of "
            "indices. image_circle_patch_area may not be accurate"
        )
    # Identify which axes of input array are spatial vs. depth dimensions
    depth_dim = img_shape.index(min(img_shape))
    spat_ind = 1 if depth_dim != 1 else 0

    # Determine which indices (along the spatial dimension) are perturbed
    pert_spatial_indices = set(np.where(x != x_adv)[spat_ind])
    if len(pert_spatial_indices) == 0:
        logger.warning("x == x_adv. image_circle_patch_area is 0")
        return 0

    # Find which indices (preceding the patch's max index) are unperturbed, in order
    # to determine the index of the edge of the patch
    max_ind_of_patch = max(pert_spatial_indices)
    unpert_ind_less_than_patch_max_ind = [
        i for i in range(max_ind_of_patch) if i not in pert_spatial_indices
    ]
    min_ind_of_patch = (
        max(unpert_ind_less_than_patch_max_ind) + 1
        if unpert_ind_less_than_patch_max_ind
        else 0
    )

    # If there are any perturbed indices outside the range of the patch just computed
    if min(pert_spatial_indices) < min_ind_of_patch:
        logger.warning("Multiple regions of the image have been perturbed")

    diameter = max_ind_of_patch - min_ind_of_patch + 1
    spatial_dims = [dim for i, dim in enumerate(img_shape) if i != depth_dim]
    patch_diameter = diameter / min(spatial_dims)
    return patch_diameter


def _generate_video_metric(metric, frame_average="mean", docstring=None):
    """
    Helper function to create video metrics from existing image metrics
    """
    mapping = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
    }
    if frame_average not in mapping:
        raise ValueError(f"frame_average {frame_average} not in {tuple(mapping)}")
    frame_average_func = mapping[frame_average]

    @numpy
    def func(x, x_adv):
        frames = []
        for x_frame, x_adv_frame in zip(x, x_adv):
            frames.append(metric(x_frame, x_adv_frame))
        return frame_average_func(frames)

    func.__doc__ = docstring
    return func


# Convenience loop for video metrics across frames
for metric_name in "l0", "l1", "l2", "linf", "image_circle_patch_diameter":
    docstring_format = "Return the {} over frames of the per-frame {} distances"
    metric = getattr(element, metric_name)
    for prefix in "mean", "max":
        docstring = docstring_format.format(prefix, metric_name)
        new_metric = _generate_video_metric(
            metric, frame_average=prefix, docstring=docstring
        )
        new_metric_name = prefix + "_" + metric_name
        if new_metric_name in globals():
            logger.warning(f"{new_metric_name} already in globals. Ignore if reloading")
        globals()[new_metric_name] = new_metric
        elementwise(new_metric, name=new_metric_name)
