"""
Test cases for perturbation metrics
"""

import pytest
import numpy as np

from armory.metrics import perturbation

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


def test_lp_norms():
    x = [1, 2, 3]
    x_adv = [2, 3, 4]
    assert (perturbation.batch.l1(x, x_adv) == np.array([1, 1, 1])).all()
    batch_size = 5
    for x, x_adv in [
        (np.ones((batch_size, 16)), np.zeros((batch_size, 16))),
        (np.ones((batch_size, 4, 4)), np.zeros((batch_size, 4, 4))),
    ]:
        for func, list_result in [
            (perturbation.batch.l2, [4.0]),
            (perturbation.batch.l1, [16.0]),
            (perturbation.batch.l0, [16.0]),
            (perturbation.batch.linf, [1.0]),
        ]:
            assert (func(x, x_adv) == np.array(list_result * batch_size)).all()
        assert (
            perturbation.batch.lp(x, x_adv, p=4) == np.array([2.0] * batch_size)
        ).all()


def test_snr():
    # variable length numpy arrays
    x = np.array(
        [
            np.array([0, 1, 0, -1]),
            np.array([0, 1, 2, 3, 4]),
        ]
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
        ]
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
