from art.attacks.evasion import (
    ProjectedGradientDescentNumpy,
    ProjectedGradientDescentPyTorch,
)
import numpy as np

from armory.logs import log


class SNR_PGDRange:
    """
    Finds the maximum SNR for each sample (if possible), via binary search.

    Example attack config:
        "attack": {
            "kwargs": {
                "batch_size": 1,
                "eps_range": [
                    10,
                    50
                ],
                "eps_step": 0.5,
                "max_iter": 10,
                "norm": "snr_db",
                "num_random_init": 2,
                "targeted": false,
                "verbose": false
            },
            "module": "armory.art_experimental.attacks.snr_pgd",
            "name": "SNR_PGDRange",
            "use_label": true
        }
    """

    def __init__(self, estimator, eps_range=(0, 10, 20, 30, 40, 50), **kwargs):
        if "eps" in kwargs:
            raise ValueError("Use 'eps_range' instead of 'eps'")
        self.estimator = estimator
        self.eps_range = sorted(eps_range)
        if len(eps_range) < 2:
            raise ValueError("Please select multiple values for eps_range")
        self.attacks = []
        for eps in eps_range:
            self.attacks.append(SNR_PGD(estimator, eps=eps, **kwargs))

    def generate(self, x, y=None, **kwargs):
        if y is None:
            raise ValueError("This attack requires given labels")
        if self.estimator.predict(x).argmax() != y:
            log.info("Original prediction failed. Returning original x.")
            return x

        # find best eps via bisection
        i_min = 0
        i_max = len(self.eps_range)
        x_best = None
        eps_best = None
        while i_min < i_max:
            i_mid = (i_min + i_max) // 2
            x_adv = self.attacks[i_mid].generate(x, y, **kwargs)
            eps = self.eps_range[i_mid]
            if self.estimator.predict(x_adv).argmax() != y:
                # attack success (will also succeed for lower SNR)
                log.info(f"Success with eps {eps}")
                x_best = x_adv
                eps_best = eps
                i_min = i_mid + 1
            else:
                # attack failure (will also fail for higher SNR)
                log.info(f"Failure with eps {eps}")
                i_max = i_mid

        if x_best is None:
            log.info("Attack failed. Returning original x.")
            return x
        if x_best is not None:
            log.info(f"Returning best attack with eps {eps_best}")
            return x_best


class SNR_PGDRange2:
    """
    Finds the maximum SNR for each sample, with a given tolerance

    Example attack config:
        "attack": {
            "knowledge": "white",
            "kwargs": {
                "batch_size": 1,
                "eps_min": 0,
                "eps_max": 64,
                "eps_step": 0.5,
                "max_iter": 10,
                "norm": "snr_db",
                "num_random_init": 2,
                "targeted": false,
                "tolerance": 1,
                "verbose": false
            },
            "module": "armory.art_experimental.attacks.snr_pgd",
            "name": "SNR_PGDRange2",
            "use_label": true
        }
    """

    def __init__(
        self, estimator, attack="l2", eps_min=10, eps_max=50, tolerance=1, **kwargs
    ):
        if eps_min > eps_max:
            raise ValueError(f"eps_min {eps_min} > eps_max {eps_max}")
        if "eps" in kwargs:
            raise ValueError("Use 'eps_min' and 'eps_max' instead of 'eps'")
        if eps_min == -np.inf or eps_max == np.inf:
            # needed due to bisection approach
            raise ValueError(f"eps_min {eps_min} and eps_max {eps_max} must be finite")

        if attack == "l2":
            self.Attack = SNR_PGD
        elif attack == "linf":
            self.Attack = SNR_PGD_Linf
        else:
            raise ValueError(f'attack {attack} not in ("l2", "linf")')
        self.min_attack = self.Attack(estimator, eps=eps_min, **kwargs)
        self.max_attack = self.Attack(estimator, eps=eps_max, **kwargs)
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.estimator = estimator
        self.kwargs = kwargs
        tolerance = float(tolerance)
        if not (tolerance >= 0):
            raise ValueError(f"tolerance {tolerance} must be a positive float")
        if tolerance == 0:
            log.warning("Using minimum float tolerance instead of 0")

        self.tolerance = tolerance

    def generate(self, x, y=None, **kwargs):
        if y is None:
            raise ValueError("This attack requires given labels")
        if self.estimator.predict(x).argmax() != y:
            log.info("Original prediction failed. Returning original x.")
            return x
        if self.eps_min == self.eps_max:
            return self.min_attack.generate(x, y, **kwargs)

        # test endpoints
        x_adv = self.max_attack.generate(x, y, **kwargs)
        if self.estimator.predict(x_adv).argmax() != y:
            log.info(f"Success at upper boundary eps = {self.eps_max}")
            return x_adv
        else:
            log.info(f"Failure at upper boundary eps = {self.eps_max}")
            upper_eps = self.eps_max

        x_adv = self.min_attack.generate(x, y, **kwargs)
        if self.estimator.predict(x_adv).argmax() != y:
            log.info(f"Success at lower boundary eps = {self.eps_min}")
            lower_eps = self.eps_min
            x_best = x_adv
            eps_best = lower_eps
        else:
            log.info(f"Failure at lower boundary eps = {self.eps_min}")
            return x_adv

        while upper_eps - lower_eps > self.tolerance:
            mid_eps = (upper_eps + lower_eps) / 2
            if mid_eps == lower_eps or mid_eps == upper_eps:
                log.info("Reached floating point tolerance limit")
                break
            attack = self.Attack(self.estimator, eps=mid_eps, **self.kwargs)
            x_adv = attack.generate(x, y, **kwargs)
            if self.estimator.predict(x_adv).argmax() != y:
                log.info(f"Success at eps = {mid_eps}")
                lower_eps = mid_eps
                x_best = x_adv
                eps_best = lower_eps
            else:
                log.info(f"Failure at eps = {mid_eps}")
                upper_eps = mid_eps

        log.info(f"Returning best attack with eps {eps_best}")
        return x_best


class SNR_PGD_Numpy(ProjectedGradientDescentNumpy):
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
            log.warning("Input all 0. Not making any change.")
            return x
        elif self.snr_sqrt_reciprocal == 0:
            return x

        eps = x_l2 * self.snr_sqrt_reciprocal
        eps_step = eps * self.step_fraction
        self.set_params(eps=eps, eps_step=eps_step)
        return super().generate(x, y=y, **kwargs)


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
        if x.shape[1] == 0:
            log.warning("Length 0 signal. Returning original.")
            return x

        x_l2 = np.linalg.norm(x, ord=2)
        if x_l2 == 0:
            log.warning("Input all 0. Not making any change.")
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
        # Get gradient wrt loss; invert it if attack is targeted
        import art
        import torch  # lgtm [py/repeated-import]

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
                log.warning(
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


class SNR_PGD_Linf(ProjectedGradientDescentPyTorch):
    """
    Applies Linf PGD to signal based on an SNR bound defined by norm 'snr' or 'snr_db'.

    The highest signal power Linf signal is a sine wave at the Nyquist frequency
        This is used to calculate the SNR bound for Linf

    In particular, the max(linf epsilon) = L2(x) / (sqrt(n) * sqrt(min(SNR epsilon))
        This is equivalent to the RMS of the signal divided by sqrt(min(SNR epsilon))
    """

    def __init__(
        self, estimator, norm="snr", eps=10, eps_step=0.5, batch_size=1, **kwargs
    ):
        if batch_size != 1:
            raise NotImplementedError("only batch_size 1 supported")
        super().__init__(estimator, norm=np.inf, batch_size=1, **kwargs)

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
        if x.shape[1] == 0:
            log.warning("Length 0 signal. Returning original.")
            return x

        x_rms = np.linalg.norm(x, ord=2) / np.sqrt(x.shape[1])
        if x_rms == 0:
            log.warning("Input all 0. Not making any change.")
            return x
        elif self.snr_sqrt_reciprocal == 0:
            return x

        eps = x_rms * self.snr_sqrt_reciprocal
        eps_step = eps * self.step_fraction
        self.set_params(eps=eps, eps_step=eps_step)
        return super().generate(x, y=y, **kwargs)
