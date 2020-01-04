"""
Updated version of art.attacks.fast_gradient.py

Uses binary search to quickly find optimal epsilon values per test point
"""

import numpy as np

from art import attacks


class FGMBinarySearch(attacks.FastGradientMethod):
    """
    Find minimum epsilon perturbations for the given inputs

    Uses binary search, up to given tolerance.

    Uses self.eps for the maximum value to consider when searching
    Uses self.eps_step for the tolerance for result granularity
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_params(minimal=True)

    def _minimal_perturbation_binary(self, x, y):
        adv = x.copy()
        y_class = np.argmax(y)

        # Get perturbation
        perturbation = self._compute_perturbation(batch, batch_labels)

        # Get current predictions
        active_indices = np.arange(len(batch))
        current_eps = self.eps_step
        while active_indices.size > 0 and current_eps <= self.eps:
            # Adversarial crafting
            adv_batch[active_indices] = self._apply_perturbation(
                batch[active_indices],
                perturbation[active_indices],
                current_eps,
            )

            # Check for success
            adv_preds = self.classifier.predict(adv_batch)  # can we pare this down?
            adv_classes = np.argmax(adv_preds, axis=1)
            # If targeted active check to see whether we have hit the target
            if self.targeted:
                active_indices = np.where(batch_classes != adv_classes)[0]
            else:
                active_indices = np.where(batch_classes == adv_classes)[0]
            current_eps += self.eps_step

        return adv

    def _minimal_perturbation(self, x, y) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the 
        class prediction change, using binary search.
        """
        adv_x = x.copy()

        # for now, ignore batching and do individually
        for i in range(adv_x.shape[0]):
            adv_x[i] = self._minimal_perturbation_binary(x[i], y[i])

        return adv_x


### Below: rewrite of art.attacks.FastGradientMethod._minimal_perturbation function

    def _minimal_perturbation_linear_batch(self, batch, batch_labels, adv_batch=None):
        if adv_batch is None:
            adv_batch = batch.copy()
        batch_classes = np.argmax(batch_labels, axis=1)

        # Get perturbation
        perturbation = self._compute_perturbation(batch, batch_labels)

        # Get current predictions
        active_indices = np.arange(len(batch))
        current_eps = self.eps_step
        while active_indices.size > 0 and current_eps <= self.eps:
            # Adversarial crafting
            adv_batch[active_indices] = self._apply_perturbation(
                batch[active_indices],
                perturbation[active_indices],
                current_eps,
            )

            # Check for success
            adv_preds = self.classifier.predict(adv_batch)  # can we pare this down?
            adv_classes = np.argmax(adv_preds, axis=1)
            # If targeted active check to see whether we have hit the target
            if self.targeted:
                active_indices = np.where(batch_classes != adv_classes)[0]
            else:
                active_indices = np.where(batch_classes == adv_classes)[0]
            current_eps += self.eps_step

        return adv_batch

    def _minimal_perturbation_linear(self, x, y) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the 
        class prediction change, using binary search.
        """
        adv_x = x.copy()

        # Compute perturbation with implicit batching
        for start in range(0, adv_x.shape[0], self.batch_size):
            end = start + self.batch_size 
            adv_batch = self._minimal_perturbation_linear_batch(
                x[start:end],
                y[start:end],
                adv_x[start:end],
            )

        return adv_x
