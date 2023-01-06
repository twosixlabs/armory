"""
Statistical metrics
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn import cluster
from sklearn.metrics import silhouette_samples

from armory.metrics.perturbation import MetricNameSpace, set_namespace

registered = MetricNameSpace()


def register(metric, name=None):
    """
    Register a statistical metric
    """
    return set_namespace(registered, metric, name=name, set_global=True)


@register
def chi2_p_value(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    perform a chi-squared test to measure the association between
    the A flags and B flags, returning a p-value.
    """
    try:
        _, chi2_p_value, _, _ = stats.chi2_contingency(
            contingency_table, correction=False
        )
    except ValueError:
        chi2_p_value = np.nan
    return [chi2_p_value]


@register
def fisher_p_value(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    perform a Fisher exact test to measure the association between
    the A flags and B flags, returning a p-value.
    """
    _, fisher_p_value = stats.fisher_exact(contingency_table, alternative="greater")
    return [fisher_p_value]


@register
def spd(contingency_table: np.ndarray) -> List[float]:
    """
    Given a 2-x-2 contingency table of the form

                          not flagged by B   |     flagged by B
                      ---------------------------------------------
    not flagged by A |           a           |          b         |
                     |---------------------------------------------
        flagged by A |           c           |          d         |
                      ---------------------------------------------

    the Statistical Parity Difference computed by

    SPD = b / (a + b) - d / (c + d)

    is one measure of the impact being flagged by A has on being flagged by B.
    """
    numerators = contingency_table[:, 1]
    denominators = contingency_table.sum(1)
    numerators[denominators == 0] = 0  # Handle division by zero:
    denominators[denominators == 0] = 1  # 0/0 => 0/1.
    fractions = numerators / denominators
    spd = fractions[0] - fractions[1]
    return [spd]


def make_contingency_tables(
    y: np.ndarray, flagged_A: np.ndarray, flagged_B: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Given a list of class labels and two arbitrary binary flags A and B,
    for each class, produce the following 2-x-2 contingency table:

                             not flagged by B   |     flagged by B
                         ---------------------------------------------
       not flagged by A |           a           |          b         |
                        |---------------------------------------------
           flagged by A |           c           |          d         |
                         ---------------------------------------------

    For example, flag A can be whether this example was classified correctly,
    while flag B reports some other binary characteristic of the data.

    Args:
        y (np.ndarray): The true labels (not necessarily binary)
        flagged_A (np.ndarray): The binary outputs of flag A
        flagged_B (np.ndarray): The binary outputs of flag B

    Returns:
        A map (Dict[int, np.ndarray]) of the per-class contingency tables.
    """

    y = np.array(y).astype(np.int64).flatten()
    flagged_A = np.array(flagged_A).astype(np.bool_).flatten()
    flagged_B = np.array(flagged_B).astype(np.bool_).flatten()

    if len(flagged_A) != len(y) or len(flagged_B) != len(y):
        raise ValueError(
            f"Expected arrays y, flagged_A, and flagged_B of the same length: \
            got {len(y)}, {len(flagged_A)}, and {len(flagged_B)}."
        )

    contingency_tables = {}
    for class_id in np.unique(y):

        items_flagged_A = flagged_A[y == class_id]
        items_flagged_B = flagged_B[y == class_id]

        a = (~items_flagged_A & ~items_flagged_B).sum()
        b = (~items_flagged_A & items_flagged_B).sum()
        c = (items_flagged_A & ~items_flagged_B).sum()
        d = (items_flagged_A & items_flagged_B).sum()

        table = np.array([[a, b], [c, d]])
        contingency_tables[class_id] = table

    return contingency_tables


@register
def filter_perplexity_fps_benign(
    y_clean: np.ndarray, poison_index: np.ndarray, poison_prediction: np.ndarray
) -> List[float]:
    """
    Measure one possible aspect of bias by seeing how closely the distribution of false
    positives matches the distribution of unpoisoned data.  The intuition is that bias
    depends not on how much of the poison the filter finds, but only what the filter does
    to the rest of the data.  That is, if it removes clean data, it should do so in
    proportion to class frequency, removing roughly the same fraction of each class.

    In particular, we do not expect that _everything_ removed by the filter should match
    the original class distribution, because we do not expect that the poison was applied
    evenly to the whole dataset in the first place.

    Args:
        y_clean:            The class labels of the original dataset
        poison_index:       The indices of elements in y_clean that were actually poisoned
        poison_prediction:  What the filter thinks is poison.  1 for poison, 0 for clean

    Returns:
        Normalized perplexity from fps to benign, where fps is the class frequency distribution of false positives,
        and benign is the class frequency distribution of the unpoisoned data

    """

    n_classes = len(np.unique(y_clean))
    # convert poison_index to binary vector the same length as data
    poison_inds = np.zeros_like(y_clean)
    poison_inds[poison_index.astype(np.int64)] = 1
    # benign is here defined to be the class distribution of the unpoisoned part of the data
    x_benign = y_clean[poison_inds == 0]
    x_benign = np.bincount(x_benign, minlength=n_classes)
    x_benign = x_benign / x_benign.sum()
    # fps is false positives: clean data marked as poison by the filter
    fp_inds = (1 - poison_inds) & poison_prediction
    fp_labels = y_clean[fp_inds == 1]
    fps = np.bincount(fp_labels, minlength=n_classes)
    if fps.sum() == 0:
        return [1]  # If no FPs, we'll define perplexity to be 1 (unbiased)
    fps = fps / fps.sum()

    return perplexity(fps, x_benign)


@register
def perplexity(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> List[float]:
    """
    Return the normalized p-to-q perplexity.
    """
    kl_div_pq = kl_div(p, q, eps)[0]
    perplexity_pq = np.exp(-kl_div_pq)
    return [perplexity_pq]


@register
def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> List[float]:
    """
    Return the Kullback-Leibler divergence from p to q.
    """
    cross_entropy_pq = cross_entropy(p, q, eps)
    entropy_p = cross_entropy(p, p, eps)
    kl_div_pq = cross_entropy_pq - entropy_p
    return [kl_div_pq]


@register
def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Return the cross entropy from a distribution p to a distribution q.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if p.ndim > 2 or q.ndim > 2:
        raise ValueError(
            f"Not obvious how to reshape arrays: got shapes {p.shape} and {q.shape}."
        )
    elif (p.ndim == 2 and p.shape[0] > 1) or (q.ndim == 2 and q.shape[0] > 1):
        raise ValueError(
            f"Expected 2-dimensional arrays to have shape (1, *): got shapes \
             {p.shape} and {q.shape}."
        )
    p = p.reshape(-1)
    q = q.reshape(-1)
    if p.shape[0] != q.shape[0]:
        raise ValueError(
            f"Expected arrays of the same length: got lengths {len(p)} and {len(q)}."
        )
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Arrays must both be non-negative.")
    if np.isclose(p.sum(), 0) or np.isclose(q.sum(), 0):
        raise ValueError("Arrays must both be non-zero.")
    if not np.isclose(p.sum(), 1):
        p /= p.sum()
    if not np.isclose(q.sum(), 1):
        q /= q.sum()
    cross_entropy_pq = (-p * np.log(q + eps)).sum()
    return cross_entropy_pq


@register
def class_bias(y_true, majority_mask, kept_mask, class_labels):
    """
    Return dict mapping from class_label -> (chi2, spd)
        chi2, spd = result[class_label]
        chi2 - chi squared statistic
        spd - statistical parity differnce
    """
    majority_contingency_tables = make_contingency_tables(
        y_true,
        majority_mask,
        kept_mask,
    )

    chi2_spd = {}
    for c in class_labels:
        class_c_table = majority_contingency_tables.get(c)
        if class_c_table is None:
            chi2_val = None
            spd_val = None
        else:
            chi2_val = np.mean(chi2_p_value(class_c_table))
            spd_val = np.mean(spd(class_c_table))
        chi2_spd[c] = (chi2_val, spd_val)

    return chi2_spd


@register
def majority_mask(
    activations,
    majority_ceiling=None,
    random_state=42,
):
    """
    Return a boolean mask and silhouette score ceiling
    """
    clusterer = cluster.KMeans(n_clusters=2, random_state=random_state)
    cluster_labels = clusterer.fit_predict(activations)
    silhouette_scores = silhouette_samples(activations, cluster_labels)
    if majority_ceiling is None:
        majority_ceiling = silhouette_scores.mean()
    mask = (0 <= silhouette_scores) & (silhouette_scores <= majority_ceiling)
    return mask, majority_ceiling


@register
def class_majority_mask(
    activations,
    class_ids,
    majority_ceilings: Optional[Dict[int, float]] = None,
    random_state: int = 42,
):
    """
    Return a single boolean mask and a silhouette score ceiling for each class
    """
    full_majority_mask = np.empty(len(activations), dtype=bool)
    if majority_ceilings is None:
        majority_ceilings = {}

    for class_id in np.unique(class_ids):
        mask_index = class_ids == class_id
        mask, majority_ceiling_id = majority_mask(
            activations[mask_index],
            majority_ceiling=majority_ceilings.get(class_id),
            random_state=random_state,
        )
        majority_ceilings[class_id] = majority_ceiling_id
        full_majority_mask[mask_index] = mask
    return full_majority_mask, majority_ceilings
