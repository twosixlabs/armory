from art.attacks.evasion import AutoAttack

from armory.utils.config_loading import load_attack


class CascadingAttack(AutoAttack):
    def __init__(self, estimator, **kwargs):
        self._check_kwargs(kwargs)
        self.targeted = kwargs.get("targeted", False)
        self.attacks = []
        for inner_config in kwargs["inner_configs"]:
            inner_config["kwargs"]["targeted"] = self.targeted
            self.attacks.append(load_attack(inner_config, estimator))
        kwargs.pop("inner_configs")
        super().__init__(estimator=estimator, attacks=self.attacks, **kwargs)

    def _check_kwargs(self, kwargs):
        if "inner_configs" not in kwargs:
            raise ValueError("Missing 'inner_configs' key in attack kwargs")
        if not isinstance(kwargs["inner_configs"], (list, tuple)):
            raise ValueError("`inner_configs` key must map to a tuple or list")
        for i, config in enumerate(kwargs["inner_configs"]):
            if "module" not in config:
                raise ValueError(f"Missing 'module' key in inner_configs[{i}]")
            if "name" not in config:
                raise ValueError(f"Missing 'name' key in inner_configs[{i}]")
