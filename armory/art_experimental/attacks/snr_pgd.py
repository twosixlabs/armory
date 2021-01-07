from art.attacks.evasion import ProjectedGradientDescentPyTorch
import numpy as np

import logging

logger = logging.getLogger(__name__)


class SNR_PGD(ProjectedGradientDescentPyTorch):
    """
    Applies L2 PGD to signal based on an SNR bound defined by norm 'snr' or 'snr_db'.
        This is a *lower* bound on allowable SNR (as opposed to L2 upper bound)

    norm - snr or snr_db
    eps - the lower bound on allowable SNR and SNR_DB
    eps_step - float value in (0, 1] that is the ratio of max L2 distance per step
        NOTE: this is different from original PGD because SNR is not additive

    SNR measures the original signal power to adversarial perturbation power
        If there is no perturbation, SNR is np.inf; all other values are finite.
        If the perturbation is unbounded, SNR is 0
    SNR_DB = 10 * log10(SNR)

    If SNR is set to 0 or SNR_DB to -inf, no limit is provided to PGD
        If SNR or SNR_DB is set to inf, no perturbation is performed
    """

    def __init__(
        self, estimator, norm="snr", eps=10, eps_step=0.5, batch_size=1, **kwargs
    ):
        if batch_size != 1:
            raise NotImplementedError("only batch_size 1 supported")
        super().__init__(estimator, norm=2, batch_size=1, **kwargs)

        # Map to SNR domain
        eps = float(eps)
        if norm == "snr":
            snr = eps
        elif norm == "snr_db":
            snr = 10 ** (eps / 10)
        else:
            raise ValueError(f"norm must be 'snr' (default) or 'snr_db', not {norm}")

        if snr < 0:
            raise ValueError(f"snr must be nonnegative, not {snr}")
        elif snr == 0:
            self.snr_sqrt_reciprocal = np.inf
        elif snr == np.inf:
            self.snr_sqrt_reciprocal = 0
        else:
            self.snr_sqrt_reciprocal = 1 / np.sqrt(snr)

        eps_step = float(eps_step)

        if not (0 < eps_step <= 1):
            raise ValueError(f"eps_step must be in (0, 1], not {eps_step}")
        self.step_fraction = eps_step

    def generate(self, x, y=None, **kwargs):
        x_l2 = np.linalg.norm(x, ord=2)
        if x_l2 == 0:
            logger.warning("Input all 0. Not making any change.")
            return x
        elif self.snr_sqrt_reciprocal == 0:
            return x

        eps = x_l2 * self.snr_sqrt_reciprocal
        eps_step = eps * self.step_fraction
        self.set_params(eps=eps, eps_step=eps_step)
        return super().generate(x, y=y, **kwargs)

    def _compute_perturbation(self, x, y, mask):
        """
        Compute perturbations.
        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Get gradient wrt loss; invert it if attack is targeted
        import art

        if art.__version__.startswith("1.4"):
            grad = self.estimator.loss_gradient_framework(x, y) * (
                1 - 2 * int(self.targeted)
            )
        else:
            grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))
        assert x.shape == grad.shape

        # Apply mask
        if mask is not None:
            grad = grad * mask

        # Remove NaNs
        grad[torch.isnan(grad)] = 0.0
        return grad

    def _apply_perturbation(self, x, grad, eps_step):
        """
        Apply perturbation on examples.
        :param x: Current adversarial examples.
        :param grad: Current gradient.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Apply norm bound
        ind = tuple(range(1, len(x.shape)))

        normalization = (
            torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol
        )
        # eps_step > 0, normalization > 0; either could be inf
        if eps_step == float("inf"):  # ignore normalization
            perturbation = grad * eps_step
        else:
            if not torch.isfinite(normalization).all():
                logger.warning(
                    "Some gradient values are infinite. Perturbation will be 0!"
                )
            perturbation = grad * eps_step / normalization
        perturbation[torch.isnan(perturbation)] = 0.0

        x = x + perturbation
        x[torch.isnan(perturbation)] = 0.0

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x
