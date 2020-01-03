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

    def _minimal_perturbation_batch(self, batch, batch_labels):
        # Get perturbation
        perturbation = self._compute_perturbation(batch, batch_labels)

        # Get current predictions
        active_indices = np.arange(len(batch))
        current_eps = self.eps_step
        while active_indices.size > 0 and current_eps <= self.eps:
            # Adversarial crafting
            current_x = self._apply_perturbation(x[batch_index_1:batch_index_2], perturbation, current_eps)
            # Update
            batch[active_indices] = current_x[active_indices]
            adv_preds = self.classifier.predict(batch)
            # If targeted active check to see whether we have hit the target, otherwise head to anything but
            if self.targeted:
                active_indices = np.where(np.argmax(batch_labels, axis=1) != np.argmax(adv_preds, axis=1))[0]
            else:
                active_indices = np.where(np.argmax(batch_labels, axis=1) == np.argmax(adv_preds, axis=1))[0]

            current_eps += self.eps_step

    def _minimal_perturbation(self, x, y) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the 
        class prediction change, using binary search.
        """
        adv_x = x.copy()

        # Compute perturbation with implicit batching
        for start in range(0, adv_x.shape[0], self.batch_size):
            end = start + self.batch_size 
            adv_batch = self._minimal_perturbation_batch(adv_x[start:end], y[start:end])
            adv_x[start:end] = adv_batch

        return adv_x
