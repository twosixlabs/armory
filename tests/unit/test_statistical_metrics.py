"""
Test cases for statistical metrics
"""


import numpy as np
import pytest

from armory.metrics import statistical

pytestmark = pytest.mark.unit


def test_chi2_p_value():

    table1 = np.array([[2, 3], [4, 6]])
    table2 = np.array([[5, 1], [1, 5]])
    assert statistical.chi2_p_value(table1)[0] == pytest.approx(1)
    assert statistical.chi2_p_value(table2)[0] == pytest.approx(0.02092134)


def test_fisher_p_value():

    table1 = np.array([[2, 3], [4, 6]])
    table2 = np.array([[5, 1], [1, 5]])
    assert statistical.fisher_p_value(table1)[0] == pytest.approx(0.7062937)
    assert statistical.fisher_p_value(table2)[0] == pytest.approx(0.04004329)


def test_spd():
    table1 = np.array([[2, 3], [4, 6]])
    table2 = np.array([[5, 1], [1, 5]])
    assert statistical.spd(table1)[0] == pytest.approx(0)
    assert statistical.spd(table2)[0] == pytest.approx(-0.6666667)


def test_make_contingency_tables():
    y = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
        ]
    )
    flagA = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    flagB = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])

    tables = {
        0: np.array([[1, 1], [1, 0]]),
        1: np.array([[1, 0], [1, 1]]),
        2: np.array([[2, 0], [0, 1]]),
    }

    tables_ = statistical.make_contingency_tables(y, flagA, flagB)
    for c in np.unique(y):
        assert tables[c] == pytest.approx(tables_[c])


def test_filter_perplexity_fps_benign():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    poison = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    poison_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    poison_inds = np.where(poison)[0]
    assert statistical.filter_perplexity_fps_benign(y, poison_inds, poison_pred)[
        0
    ] == pytest.approx(0.983360398668428)

    poison_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert statistical.filter_perplexity_fps_benign(y, poison_inds, poison_pred)[
        0
    ] == pytest.approx(0.416666666725)


def test_perplexity():
    p = [1, 0, 0, 0]
    q = [0.9, 0.1, 0, 0]
    assert statistical.perplexity(p, q)[0] == pytest.approx(0.9)

    p = [0.25, 0.25, 0.25, 0.25]
    assert statistical.perplexity(p, q)[0] == pytest.approx(2.190890229752889e-05)


def test_kl_div():
    p = [1, 0, 0, 0]
    q = [0.9, 0.1, 0, 0]
    assert statistical.kl_div(p, q)[0] == pytest.approx(0.10536051565782628)

    p = [0.25, 0.25, 0.25, 0.25]
    assert statistical.kl_div(p, q)[0] == pytest.approx(10.728617506135528)


def test_cross_entropy():
    p = [1, 0, 0, 0]
    q = [0.9, 0.1, 0, 0]
    assert statistical.cross_entropy(p, q) == pytest.approx(0.10536051554671516)

    p = [0.25, 0.25, 0.25, 0.25]
    assert statistical.cross_entropy(p, q) == pytest.approx(12.114911866855419)


def test_class_bias():
    class_labels = [0, 1, 2]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    flagA = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    flagB = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]

    chi2_spd = statistical.class_bias(y, flagA, flagB, class_labels)
    answer = {
        0: (0.7093881150142265, -0.16666666666666663),
        1: (0.7093881150142265, 0.16666666666666669),
        2: (0.3613104285261789, 0.33333333333333337),
    }
    for key in class_labels:
        assert chi2_spd[key] == pytest.approx(answer[key])

    flagA = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    chi2_spd = statistical.class_bias(y, flagA, flagB, class_labels)
    answer = {
        0: (0.025347318677468325, -1.0),
        1: (0.025347318677468325, -1.0),
        2: (0.17090352023079353, -0.5),
    }
    for key in class_labels:
        assert chi2_spd[key] == pytest.approx(answer[key])


def test_majority_mask():

    activations = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
            [1.1, 1.1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-0.9, -0.9],
        ]
    )

    mask, ceiling = statistical.majority_mask(activations)
    assert (mask == [False, False, False, True, False, False, False, True]).all()
    assert ceiling == pytest.approx(0.9749804436907279)


def test_class_majority_mask():

    activations = np.array(
        [
            [1, 1],
            [1, 1],
            [1.1, 1.1],
            [-1, -1],
            [-1, -1],
            [-0.9, -0.9],
            [1, 1],
            [1, 2],
            [2, 1],
            [-1, 1],
            [-1, 2],
            [-2, 1],
        ]
    )

    labels = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    true_ceilings = {0: 0.9666434966331335, 1: 0.5798612907787333}
    true_mask = [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
    ]

    mask, ceilings = statistical.class_majority_mask(activations, labels)
    assert (mask == true_mask).all()
    assert ceilings == pytest.approx(true_ceilings)
