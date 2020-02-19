import json
from typing import Optional


def _verify_adhoc(config: Optional[dict]):
    if config is None:
        return
    else:
        assert isinstance(config, dict)


def _verify_attack(config: Optional[dict]):
    if config is None:
        return
    else:
        assert isinstance(config, dict)
        if "module" not in config.keys():
            raise ValueError("A `module` must be specified in attack json")
        assert isinstance(config["module"], str)

        if "name" not in config.keys():
            raise ValueError("A `name` must be specified in attack json")
        assert isinstance(config["name"], str)

        if "kwargs" not in config.keys():
            raise ValueError("`kwargs` must be specified in attack json")
        if config["kwargs"] is not None:
            assert isinstance(config["kwargs"], dict)

        if "knowledge" not in config.keys():
            raise ValueError("`knowledge` must be specified in attack json")
        if config["knowledge"] is not None:
            assert isinstance(config["knowledge"], str)

        if "budget" not in config.keys():
            raise ValueError("`budget` field must contain a nested json")
        if config["budget"] is not None:
            assert isinstance(config["budget"], dict)


def _verify_dataset(config: Optional[dict]):
    if config is None:
        return
    else:
        if "module" not in config.keys():
            raise ValueError("A `module` must be specified in dataset json")
        assert isinstance(config["module"], str)

        if "name" not in config.keys():
            raise ValueError("A `name` must be specified in dataset json")
        assert isinstance(config["name"], str)


def _verify_defense(config: Optional[dict]):
    if config is None:
        return
    else:
        assert isinstance(config, dict)

        if "module" not in config.keys():
            raise ValueError("A `module` must be specified in defense json")
        assert isinstance(config["module"], str)

        if "name" not in config.keys():
            raise ValueError("A `name` must be specified in defense json")
        assert isinstance(config["name"], str)

        if "kwargs" not in config.keys():
            raise ValueError("`kwargs` must be specified in defense json")
        if config["kwargs"] is not None:
            assert isinstance(config["kwargs"], dict)


def _verify_evaluation(config: Optional[dict]):
    if config is None:
        raise ValueError("Evaluation field must contain a nested json")
    else:
        assert isinstance(config, dict)

        if "eval_file" not in config.keys():
            raise ValueError("An `eval_file` must be specified in evaluation json")
        assert isinstance(config["eval_file"], str)


def _verify_metric(config: Optional[dict]):
    if config is None:
        return
    else:
        assert isinstance(config, dict)

        if "module" not in config.keys():
            raise ValueError("A `module` must be specified in metric json")
        assert isinstance(config["module"], str)

        if "name" not in config.keys():
            raise ValueError("A `name` must be specified in metric json")
        assert isinstance(config["name"], str)

        if "kwargs" not in config.keys():
            raise ValueError("`kwargs` must be specified in metric json")
        if config["kwargs"] is not None:
            assert isinstance(config["kwargs"], dict)


def _verify_model(config: Optional[dict]):
    if config is None:
        return
    else:
        assert isinstance(config, dict)

        if "module" not in config.keys():
            raise ValueError("A `module` must be specified in model json")
        assert isinstance(config["module"], str)

        if "name" not in config.keys():
            raise ValueError("A `name` must be specified in model json")
        assert isinstance(config["name"], str)

        if "model_kwargs" not in config.keys():
            raise ValueError("`model_kwargs` must be specified in model json")
        assert isinstance(config["model_kwargs"], dict)

        if "wrapper_kwargs" not in config.keys():
            raise ValueError("`wrapper_kwargs` must be specified in model json")
        assert isinstance(config["wrapper_kwargs"], dict)


def _verify_sysconfig(config: Optional[dict]):
    if config is None:
        raise ValueError("sysconfig field must contain a nested json")
    else:
        assert isinstance(config, dict)

        if "docker_image" not in config.keys():
            raise ValueError("A `docker_image` must be specified in sysconfig json")
        assert isinstance(config["docker_image"], str)

        if "external_github_repo" not in config.keys():
            raise ValueError(
                "An `external_github_repo` field must exist in sysconfig json"
            )
        if config["external_github_repo"] is not None:
            assert isinstance(config["external_github_repo"], str)

        if "use_gpu" not in config.keys():
            raise ValueError("A `use_gpu` boolean must be specified in sysconfig json")
        assert isinstance(config["use_gpu"], bool)


def verify_config(config: dict) -> dict:
    for k in (
        "adhoc",
        "attack",
        "dataset",
        "defense",
        "evaluation",
        "metric",
        "model",
        "sysconfig",
    ):
        if k not in config.keys():
            raise ValueError(f"{k} is missing from configuration file.")

        _verify_adhoc(config["adhoc"])
        _verify_attack(config["attack"])
        _verify_dataset(config["dataset"])
        _verify_defense(config["defense"])
        _verify_evaluation(config["evaluation"])
        _verify_metric(config["metric"])
        _verify_model(config["model"])
        _verify_sysconfig(config["sysconfig"])

    return config


def load_config(filepath: str) -> dict:
    with open(filepath) as f:
        config = json.load(f)

    return verify_config(config)
