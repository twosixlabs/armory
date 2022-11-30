"""
Updated version of art.attacks.fast_gradient.py

Uses binary search to quickly find optimal epsilon values per test point
"""

from art.attacks.evasion import FastGradientMethod
import numpy as np


class FGMBinarySearch(FastGradientMethod):
    """
    Find minimum epsilon perturbations for the given inputs

    Uses binary search, up to given tolerance.

    Uses self.eps for the maximum value to consider when searching
    Uses self.eps_step for the tolerance for result granularity
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_params(minimal=True)

    def _minimal_perturbation_binary_batch(self, batch, batch_labels, adv_batch=None):
        """
        Iteratively compute the minimal perturbation necessary to make the
        class prediction change, using binary search.

        batch - np array of features for batch (x)
        batch_labels - np array of labels for batch, a 2D array of probabilites (y)
        adv_batch - same shape as batch (x^hat)
            This method returns the adv_batch that is computed for the given x and y
            If adv_batch is None, it creates a new array to return
            Else, it modifies the existing array that is passed in

        Returns - computed adv_batch
        """
        if adv_batch is None:
            adv_batch = batch.copy()
        batch_classes = np.argmax(batch_labels, axis=1)

        # Get perturbation
        mask = None
        perturbation = self._compute_perturbation(batch, batch_labels, mask)

        def check_epsilon(i, epsilon):
            adv_batch = self._apply_perturbation(batch[[i]], perturbation[[i]], epsilon)
            adv_pred = self.estimator.predict(adv_batch)
            adv_class = np.argmax(adv_pred, axis=1)
            if self.targeted:
                success = batch_classes[[i]] == adv_class
            else:
                success = batch_classes[[i]] != adv_class
            return success, adv_batch

        tolerance = self.eps_step
        for i in range(len(batch)):
            # Assume endpoints are correct
            min_eps = 0
            max_eps = self.eps
            while max_eps - min_eps > tolerance:
                mid_eps = (max_eps + min_eps) / 2
                # print(min_eps, mid_eps, max_eps, tolerance)

                success, adv_i = check_epsilon(i, mid_eps)
                if success:
                    adv_batch[[i]] = adv_i
                    max_eps = mid_eps
                else:
                    min_eps = mid_eps

        return adv_batch

    def _minimal_perturbation_linear_batch(self, batch, batch_labels, adv_batch=None):
        """
        Rewrite of inner loop for linear search

        batch - np array of features for batch (x)
        batch_labels - np array of labels for batch, a 2D array of probabilites (y)
        adv_batch - same shape as batch (x^hat)
            This method returns the adv_batch that is computed for the given x and y
            If adv_batch is None, it creates a new array to return
            Else, it modifies the existing array that is passed in

        Returns - computed adv_batch
        """
        if adv_batch is None:
            adv_batch = batch.copy()
        batch_classes = np.argmax(batch_labels, axis=1)
        adv_classes = batch_classes.copy()

        # Get perturbation
        perturbation = self._compute_perturbation(batch, batch_labels)

        # Get current predictions
        active = np.arange(len(batch))
        current_eps = self.eps_step
        while active.size > 0 and current_eps <= self.eps:
            # Adversarial crafting
            adv_batch[active] = self._apply_perturbation(
                batch[active],
                perturbation[active],
                current_eps,
            )

            # Check for success
            adv_preds = self.estimator.predict(adv_batch[active])
            # adv_preds = self.estimator.predict(adv_batch)  # can we pare this down?
            adv_classes[active] = np.argmax(adv_preds, axis=1)
            # If targeted active check to see whether we have hit the target
            if self.targeted:
                active = np.where(batch_classes != adv_classes)[0]
            else:
                active = np.where(batch_classes == adv_classes)[0]
            current_eps += self.eps_step

        return adv_batch

    def _minimal_perturbation(self, x, y, mask) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the
        class prediction change, using binary search.
        """
        if mask is not None:
            raise NotImplementedError("non-None mask not implemented")
        adv_x = x.copy()

        # Compute perturbation with implicit batching
        for start in range(0, adv_x.shape[0], self.batch_size):
            end = start + self.batch_size
            self._minimal_perturbation_binary_batch(
                x[start:end],
                y[start:end],
                adv_x[start:end],
            )

        return adv_x
