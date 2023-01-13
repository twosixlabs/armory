"""
Test cases for perturbation metrics
"""

import itertools

import numpy as np
import pytest

from armory.metrics import perturbation

pytestmark = pytest.mark.unit


def test_lp_norms():
    x = [1, 2, 3]
    x_adv = [2, 3, 4]
    assert (perturbation.batch.l1(x, x_adv) == np.array([1, 1, 1])).all()
    for x_i, x_adv_i in zip(x, x_adv):
        assert perturbation.element.l1(x_i, x_adv_i) == 1
    batch_size = 5
    for x, x_adv in [
        (np.ones((batch_size, 16)), np.zeros((batch_size, 16))),
        (np.ones((batch_size, 4, 4)), np.zeros((batch_size, 4, 4))),
    ]:
        for batch_func, element_func, list_result in [
            (perturbation.batch.l2, perturbation.element.l2, [4.0]),
            (perturbation.batch.l1, perturbation.element.l1, [16.0]),
            (perturbation.batch.l0, perturbation.element.l0, [16.0]),
            (perturbation.batch.linf, perturbation.element.linf, [1.0]),
        ]:
            assert (batch_func(x, x_adv) == np.array(list_result * batch_size)).all()
            assert element_func(x[0], x_adv[0]) == list_result[0]
        assert (
            perturbation.batch.lp(x, x_adv, p=4) == np.array([2.0] * batch_size)
        ).all()
        assert perturbation.element.lp(x[0], x_adv[0], p=4) == 2.0


def test_snr():
    # variable length numpy arrays
    x = np.array(
        [
            np.array([0, 1, 0, -1]),
            np.array([0, 1, 2, 3, 4]),
        ],
        dtype=object,
    )

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 4),
        (1, np.inf),
        (2, 1),
        (3, 0.25),
        (11, 0.01),
    ]:
        assert (perturbation.batch.snr(x, x * multiplier) == [snr_value] * len(x)).all()
        assert (
            perturbation.batch.snr_db(x, x * multiplier)
            == [10 * np.log10(snr_value)] * len(x)
        ).all()

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        assert (perturbation.batch.snr(x, x + addition) == [snr_value] * len(x)).all()
        assert (
            perturbation.batch.snr_db(x, x + addition)
            == [10 * np.log10(snr_value)] * len(x)
        ).all()

    with pytest.raises(ValueError):
        perturbation.batch.snr(x[:1], x[1:])
    with pytest.raises(ValueError):
        perturbation.batch.snr(x, np.array([1]))


def test_snr_spectrogram():
    # variable length numpy arrays
    x = np.array(
        [
            np.array([0, 1, 0, -1]),
            np.array([0, 1, 2, 3, 4]),
        ],
        dtype=object,
    )

    for multiplier, snr_value in [
        (0, 1),
        (0.5, 2),
        (1, np.inf),
        (2, 1),
        (3, 0.5),
        (11, 0.1),
    ]:
        assert (
            perturbation.batch.snr_spectrogram(x, x * multiplier)
            == [snr_value] * len(x)
        ).all()
        assert (
            perturbation.batch.snr_spectrogram_db(x, x * multiplier)
            == [10 * np.log10(snr_value)] * len(x)
        ).all()

    for addition, snr_value in [
        (0, np.inf),
        (np.inf, 0.0),
    ]:
        assert (
            perturbation.batch.snr_spectrogram(x, x + addition) == [snr_value] * len(x)
        ).all()
        assert (
            perturbation.batch.snr_spectrogram_db(x, x + addition)
            == [10 * np.log10(snr_value)] * len(x)
        ).all()

    with pytest.raises(ValueError):
        perturbation.batch.snr_spectrogram(x[:1], x[1:])
    with pytest.raises(ValueError):
        perturbation.batch.snr_spectrogram(x, np.array([1]))


def test_image_circle_patch_diameter(caplog):
    image_circle_patch_diameter = perturbation.image_circle_patch_diameter
    with pytest.raises(ValueError):
        image_circle_patch_diameter([1, 1, 1], [1, 1, 1])
    with pytest.raises(ValueError):
        image_circle_patch_diameter(np.ones((3, 3, 3, 3)), np.ones((3, 3, 3, 3)))

    x = np.zeros((28, 28))
    assert image_circle_patch_diameter(x, x) == 0.0

    assert image_circle_patch_diameter(x, x + 1) == 1.0
    assert "x and x_adv differ at 100% of indices." in caplog.text

    x = np.zeros((100, 10, 1))
    assert image_circle_patch_diameter(x, x + 1) == 10.0
    assert "Circular patch is not contained within the image" in caplog.text
    assert image_circle_patch_diameter(x.transpose(), x.transpose() + 1) == 10.0

    N = 10
    x = np.zeros((N, N))
    x_adv = np.zeros((N, N))
    # Draw widening circles centered at (5, 5)
    for r in range(5):
        for i, j in itertools.product(range(x.shape[0]), range(x.shape[1])):
            if (i - 5) ** 2 + (j - 5) ** 2 <= r**2:
                x_adv[i, j] = 1
        assert image_circle_patch_diameter(x, x_adv) == (r * 2 + 1) / N
        for axis in range(3):
            assert (
                image_circle_patch_diameter(
                    np.expand_dims(x, axis=axis), np.expand_dims(x_adv, axis=axis)
                )
                == (r * 2 + 1) / N
            )


def test_video_metrics():
    def func(x, x_adv):
        return (x + x_adv).sum()

    with pytest.raises(ValueError):
        perturbation._generate_video_metric(func, frame_average="not right")

    docstring = "My Custom Docstring"
    image_dim = (28, 28, 1)
    pixels = np.product(image_dim)
    for name in "mean", "max", "min":
        new_metric = perturbation._generate_video_metric(
            func, frame_average=name, docstring=docstring
        )
        assert new_metric.__doc__ == docstring

    x = np.zeros((3,) + image_dim)
    x_adv = np.stack(
        [0 * np.ones(image_dim), 1 * np.ones(image_dim), 2 * np.ones(image_dim)]
    )

    assert perturbation.element.max_l1(x, x_adv) == 2 * pixels
    assert perturbation.element.mean_l1(x, x_adv) == pixels
    min_l1 = perturbation._generate_video_metric(
        perturbation.element.l1, frame_average="min"
    )
    assert min_l1(x, x_adv) == 0
