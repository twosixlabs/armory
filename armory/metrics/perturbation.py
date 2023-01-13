"""
Metrics to measure adversarial perturbations
"""

import functools

import numpy as np

from armory.logs import log
from armory.metrics.common import MetricNameSpace, as_batch, set_namespace

element = MetricNameSpace()
batch = MetricNameSpace()


def batchwise(metric, name=None):
    """
    Register a batch metric and register a batchwise version of it
    """
    return set_namespace(batch, metric, name=name, set_global=True)


def elementwise(metric, name=None):
    """
    Register a element metric and register a batchwise version of it
    """
    if name is None:
        name = metric.__name__
    set_namespace(element, metric, name=name)
    batch_metric = as_batch(metric)
    batchwise(batch_metric, name=name)
    return metric


def numpy(function):
    """
    Ensures args (but not kwargs) are passed in as numpy vectors
    It casts them to to a common data type (complex, float, int) if possible
    And it ensures that they have the same shape
    """

    @functools.wraps(function)
    def wrapper(x, x_adv, **kwargs):
        x, x_adv = (np.asarray(i) for i in (x, x_adv))
        if x.shape != x_adv.shape:
            raise ValueError(f"x.shape {x.shape} != x_adv.shape {x_adv.shape}")

        # elevate to 64-bit types first to prevent overflow errors
        for umbrella_type, target_type in [
            (np.complexfloating, complex),
            (np.floating, float),
            (np.integer, int),
        ]:
            if np.issubdtype(x.dtype, umbrella_type) or np.issubdtype(
                x_adv.dtype, umbrella_type
            ):
                if x.dtype != target_type:
                    x = x.astype(target_type)
                if x_adv.dtype != target_type:
                    x_adv = x_adv.astype(target_type)
                break
        # Otherwise, do not modify

        return function(x, x_adv, **kwargs)

    return wrapper


@elementwise
@numpy
def lp(x, x_adv, *, p=2):
    """
    Return the Lp (vector) norm distance between the input arrays
        Results follow np.linalg.norm

    p - any value castable to float
    """
    return np.linalg.norm((x - x_adv).flatten(), ord=float(p))


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
    Return the L0 'norm' distance between the input arrays
        NOTE: no longer normalized by the number of elements in the array
    """
    return lp(x, x_adv, p=0)


@elementwise
@numpy
def snr(x, x_adv):
    """
    Return the absolute SNR of x to (x - x_adv), with range [0, inf]
        If there is no adversarial perturbation, always return inf
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


@elementwise
@numpy
def image_circle_patch_diameter(x, x_adv):
    """
    Return smallest circular patch diameter that covers the modified pixels
        This value is normalized by the smallest spatial dimension
        So that a value of 1 is the largest circular patch that fits inside the image
            Values greater than 1 are possible for non-square images

    This metric *assumes* that the modification was circular and contained in the image
        If the modification is non-circular, then this may underestimate diameter

    Note: If this assumption was not made, a more complicated algorithm would be needed
        See: "Smallest enclosing disks (balls and ellipsoids)", Emo Welzl, (1991)
            or https://en.wikipedia.org/wiki/Smallest-circle_problem
    """
    if x.ndim == 2:
        x = np.expand_dims(x, axis=-1)
        x_adv = np.expand_dims(x_adv, axis=-1)
    if x.ndim != 3:
        raise ValueError(
            f"Expected image with 2 (HW) or 3 (HWC or CHW) dimensions. x has shape {x.shape}"
        )

    # Assume channel dim is the smallest dimension
    channel_dim = np.argmin(x.shape)
    # mask is True if the pixel is different
    mask = (x != x_adv).any(axis=channel_dim)

    diff = mask.sum()
    if not diff:
        return 0

    # Get patch diameters for height and width, then take the larger of them
    diameters = []
    for axis in 0, 1:
        marginal = mask.any(axis=axis)
        min_index = np.argmax(marginal)
        max_index = len(marginal) - np.argmax(marginal[::-1]) - 1
        diameters.append(max_index - min_index + 1)
    diameter = max(diameters)

    size = np.product(mask.shape)
    max_diameter = min(mask.shape)  # max in-image diameter
    if diameter > max_diameter:
        log.warning("Circular patch is not contained within the image")
    elif diff >= np.pi * (max_diameter / 2) ** 2:
        log.warning(
            f"x and x_adv differ at {diff/size:.0%} of indices. "
            "Assumption of circular patch within image may be violated."
        )

    return diameter / max_diameter


# Image-based metrics applied to video
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


# Convenience loop for defining video metrics across frames
for metric_name in "l0", "l1", "l2", "linf", "image_circle_patch_diameter":
    docstring_template = "Return the {} over frames of the per-frame {} distances"
    metric = getattr(element, metric_name)
    for prefix in "mean", "max":
        docstring = docstring_template.format(prefix, metric_name)
        new_metric = _generate_video_metric(
            metric, frame_average=prefix, docstring=docstring
        )
        new_metric_name = prefix + "_" + metric_name
        globals()[new_metric_name] = new_metric
        elementwise(new_metric, name=new_metric_name)
