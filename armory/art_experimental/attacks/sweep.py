from importlib import import_module

from art.attacks import EvasionAttack
import numpy as np

from armory.logs import log


class SweepAttack(EvasionAttack):
    def __init__(self, estimator, attack_fn, sweep_params, **kwargs):
        if not isinstance(sweep_params, dict):
            raise ValueError(
                "If attack_config['type'] == 'sweep', then "
                "attack_config['sweep_params'] must be specified with "
                "'kwargs' and/or 'generate_kwargs' field(s)."
            )
        for key in sweep_params:
            if key not in ["kwargs", "generate_kwargs", "metric"]:
                raise ValueError(
                    f"Received an unexpected field '{key}' in attack_config['sweep_params']. "
                    f"The list of allowable fields is as follows: "
                    f"['kwargs', 'generate_kwargs', 'metric']."
                )
        if (
            "kwargs" not in sweep_params.keys()
            and "generate_kwargs" not in sweep_params.keys()
        ):
            raise ValueError(
                "attack_config['sweep_params'] must contain at least one "
                "of the following fields: ['kwargs', 'generate_kwargs']."
            )

        self.sweep_kwargs = sweep_params.get("kwargs", {})
        self.sweep_generate_kwargs = sweep_params.get("generate_kwargs", {})
        if len(self.sweep_kwargs) + len(self.sweep_generate_kwargs) == 0:
            raise ValueError(
                "Received no kwargs to sweep over. At least one of "
                "attack_config['sweep_params']['kwargs'] and "
                "attack_config['sweep_params']['generate_kwargs'] should be non-empty."
            )

        if len(kwargs.keys() & self.sweep_kwargs.keys()) > 0:
            log.warning(
                f"The following kwargs were set in both attack_config['kwargs'] and "
                f"attack_config['sweep_params']['kwargs']: "
                f"{list(kwargs.keys() & self.sweep_kwargs.keys())}. Values specified "
                f"in attack_config['kwargs'] will be ignored."
            )

        self._check_kwargs(self.sweep_kwargs)
        self._check_kwargs(self.sweep_generate_kwargs)
        self._load_metric_fn(sweep_params.get("metric", {}))

        self.targeted = kwargs.get("targeted", False)
        self.attacks = []
        self._estimator = estimator

        if len(self.sweep_kwargs) > 0:
            self.num_search_points = len(list(self.sweep_kwargs.values())[0])
        else:
            self.num_search_points = len(list(self.sweep_generate_kwargs.values())[0])

        for i in range(self.num_search_points):
            subattack_i_kwargs = kwargs.copy()
            for sweep_attack_kwarg in self.sweep_kwargs:
                subattack_i_kwargs[sweep_attack_kwarg] = self.sweep_kwargs[
                    sweep_attack_kwarg
                ][i]
            subattack_i = attack_fn(estimator, **subattack_i_kwargs)
            self.attacks.append(subattack_i)

    def generate(self, x, y=None, **kwargs):
        if x.shape[0] != 1:
            raise ValueError("Dataset batch_size should be set to 1")
        if y is None:
            raise ValueError(
                "This attack requires given labels. Ensure that attack['use_label'] is set to True."
            )

        if len(kwargs.keys() & self.sweep_generate_kwargs.keys()) > 0:
            log.warning(
                f"The following kwargs were set in both attack_config['generate_kwargs'] and "
                f"attack_config['sweep_params']['generate_kwargs']: "
                f"{list(kwargs.keys() & self.sweep_generate_kwargs.keys())}. Values specified "
                f"in attack_config['generate_kwargs'] will be ignored."
            )

        y_pred = self._estimator.predict(x)
        if not self._is_robust(y, y_pred):
            log.info(
                f"Estimator is not robust to original x as measured with metric "
                f"function {self.metric_fn.__name__} and threshold "
                f"{self.metric_threshold}. Returning original x."
            )
            return x

        i_min = 0
        i_max = self.num_search_points
        x_best = None
        while i_min < i_max:
            i_mid = (i_min + i_max) // 2
            attack_kwargs = {k: v[i_mid] for k, v in self.sweep_kwargs.items()}
            generate_kwargs = {
                k: v[i_mid] for k, v in self.sweep_generate_kwargs.items()
            }
            kwargs.update(generate_kwargs)
            x_adv = self.attacks[i_mid].generate(x, y, **kwargs)
            y_pred = self._estimator.predict(x_adv)
            if not self._is_robust(y, y_pred):
                # attack success
                log.info(
                    f"Success with kwargs {attack_kwargs} and generate_kwargs {generate_kwargs}"
                )
                x_best = x_adv
                best_attack_kwargs = attack_kwargs
                best_generate_kwargs = generate_kwargs
                i_max = i_mid
            else:
                # attack failure
                i_min = i_mid + 1
                log.info(
                    f"Failure with kwargs {attack_kwargs} and generate_kwargs {generate_kwargs}"
                )

        if x_best is None:
            log.info(
                "Sweep attack concluded. Returning original x since attack failed at all sweep points."
            )
            return x
        if x_best is not None:
            log.info(
                f"Sweep attack concluded. Returning the weakest-strength successful attack with "
                f"kwargs {best_attack_kwargs} and generate_kwargs {best_generate_kwargs}"
            )
            return x_best

    def _is_robust(self, y, y_pred):
        metric_result = self._get_metric_result(y, y_pred)
        if self.targeted:
            return metric_result < self.metric_threshold
        else:
            return metric_result > self.metric_threshold

    def _get_metric_result(self, y, y_pred):
        if isinstance(y, np.ndarray) and y.dtype == np.object:
            # convert np object array to list of dicts
            metric_result = self.metric_fn([y[0]], y_pred)
        else:
            metric_result = self.metric_fn(y, y_pred)
        if isinstance(metric_result, list):
            if len(metric_result) > 1:
                raise ValueError(
                    f"Expected metric function to return one value, not {len(metric_result)}."
                )
            metric_result = metric_result[0]
        elif isinstance(metric_result, np.ndarray):
            if metric_result.size > 1:
                raise ValueError(
                    f"Expected metric function to return one value, not {metric_result.size}."
                )
            metric_result = metric_result[0]
        if not isinstance(metric_result, (float, int)):
            raise TypeError(
                f"Expected metric function to return float or int, not type {type(metric_result)}."
            )
        return metric_result

    def _check_kwargs(self, kwargs):
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"Expected dict for 'sweep_params' kwargs, received type {type(kwargs)}"
            )
        num_search_points = None
        for kwarg, values in kwargs.items():
            if not isinstance(values, (tuple, list)):
                raise ValueError(
                    f"The values passed for kwarg '{kwarg}' should be a list or "
                    f"tuple, not {type(values)}"
                )
            if len(values) <= 1:
                raise ValueError(
                    f"Expected multiple values to sweep over for kwarg "
                    f"'{kwarg}', received {len(values)}: {values}."
                )
            if num_search_points is None:
                num_search_points = len(values)
                num_search_points_reference_kwarg = kwarg
            else:
                if len(values) != num_search_points:
                    raise ValueError(
                        f"All kwargs lists in attack_config['sweep_params'] "
                        f"should have the same number of values. Found {len(values)} values "
                        f"for '{kwarg}' but {num_search_points} for "
                        f"'{num_search_points_reference_kwarg}'."
                    )

            if sorted(values) != values:
                log.warning(
                    f"The values of kwarg '{kwarg}' are not sorted in ascending order. "
                    f"SweepAttack's search procedure assumes that attack strength is "
                    f"monotonically increasing."
                )

    def _load_metric_fn(self, metric_dict):
        metric_module_name = metric_dict.get("module")
        if metric_module_name is None:
            # by default use categorical accuracy to measure attack success
            from armory import metrics

            log.info(
                "Using default categorical accuracy to measure attack success "
                "since attack_config['sweep_params']['metric']['module'] is "
                "unspecified."
            )
            self.metric_fn = metrics.get("categorical_accuracy")
            self.metric_threshold = (
                0.5  # for binary metric, any x s.t. 0 < x < 1 suffices
            )
        else:
            metric_module = import_module(metric_module_name)
            metric_name = metric_dict.get("name")
            if metric_name is None:
                raise ValueError(
                    "If attack_config['sweep_params']['metric']['module'] is "
                    "specified, then attack_config['sweep_params']['metric']['name'] must "
                    "be specified as well."
                )
            self.metric_fn = getattr(metric_module, metric_name)
            self.metric_threshold = metric_dict.get("threshold")
            if self.metric_threshold is None:
                raise ValueError(
                    "If attack_config['sweep_params']['metric']['module'] is specified, then "
                    "attack_config['sweep_params']['metric']['threshold'] must be "
                    "specified as well."
                )
