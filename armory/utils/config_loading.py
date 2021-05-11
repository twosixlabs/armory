"""
Helper utilies to load things from armory configuration files.
"""

from importlib import import_module
import logging

logger = logging.getLogger(__name__)

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass
from art.attacks import Attack

try:
    from art.estimators import BaseEstimator as Classifier
except ImportError:
    logger.warning(
        "ART 1.2 support is deprecated and will be removed in ARMORY 0.11. Use ART 1.3"
    )
    from art.classifiers import Classifier
from art.defences.postprocessor import Postprocessor
from art.defences.preprocessor import Preprocessor
from art.defences.trainer import Trainer

from armory.art_experimental.attacks import patch
from armory.data.datasets import ArmoryDataGenerator, EvalGenerator
from armory.data.utils import maybe_download_weights_from_s3
from armory.utils import labels


def load(sub_config):
    module = import_module(sub_config["module"])
    fn = getattr(module, sub_config["name"])
    args = sub_config.get("args", [])
    kwargs = sub_config.get("kwargs", {})
    return fn(*args, **kwargs)


def load_fn(sub_config):
    module = import_module(sub_config["module"])
    return getattr(module, sub_config["name"])


def load_dataset(dataset_config, *args, num_batches=None, **kwargs):
    """
    Loads a dataset from configuration file

    If num_batches is None, this function will return a generator that iterates
    over the entire dataset.
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    batch_size = dataset_config["batch_size"]
    for ds_kwarg in ["index", "class_ids"]:
        if ds_kwarg not in kwargs and ds_kwarg in dataset_config:
            kwargs[ds_kwarg] = dataset_config[ds_kwarg]
    framework = dataset_config.get("framework", "numpy")
    dataset = dataset_fn(batch_size=batch_size, framework=framework, *args, **kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    if dataset_config.get("check_run"):
        return EvalGenerator(dataset, num_eval_batches=1)
    if num_batches:
        return EvalGenerator(dataset, num_eval_batches=num_batches)
    return dataset


def load_model(model_config):
    """
    Loads a model and preprocessing function from configuration file

    preprocessing_fn can be a tuple of functions or None values
        If so, it applies to training and inference separately
    """
    model_module = import_module(model_config["module"])
    model_fn = getattr(model_module, model_config["name"])
    weights_file = model_config.get("weights_file", None)
    if isinstance(weights_file, str):
        weights_path = maybe_download_weights_from_s3(
            weights_file, auto_expand_tars=True
        )
    elif isinstance(weights_file, list):
        weights_path = [
            maybe_download_weights_from_s3(w, auto_expand_tars=True)
            for w in weights_file
        ]
    elif isinstance(weights_file, dict):
        weights_path = {
            k: maybe_download_weights_from_s3(v) for k, v in weights_file.items()
        }
    else:
        weights_path = None

    model = model_fn(
        model_config["model_kwargs"], model_config["wrapper_kwargs"], weights_path
    )
    if not isinstance(model, Classifier):
        raise TypeError(f"{model} is not an instance of {Classifier}")
    if not weights_file and not model_config["fit"]:
        logger.warning(
            "No weights file was provided and the model is not configured to train. "
            "Are you loading model weights from an online repository?"
        )

    preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
    if preprocessing_fn is not None:
        if isinstance(preprocessing_fn, tuple):
            if len(preprocessing_fn) != 2:
                raise ValueError(
                    f"preprocessing tuple length {len(preprocessing_fn)} != 2"
                )
            elif not all([x is None or callable(x) for x in preprocessing_fn]):
                raise TypeError(
                    f"preprocessing_fn tuple elements {preprocessing_fn} must be None or callable"
                )
        elif not callable(preprocessing_fn):
            raise TypeError(
                f"preprocessing_fn {preprocessing_fn} must be None, tuple, or callable"
            )
    return model, preprocessing_fn


def load_attack(attack_config, classifier):
    if attack_config.get("type") == "patch":
        original_kwargs = attack_config.pop("kwargs")
        kwargs = original_kwargs.copy()
        apply_patch_args = kwargs.pop("apply_patch_args", [])
        apply_patch_kwargs = kwargs.pop("apply_patch_kwargs", {})
        targeted = kwargs.pop("targeted", None)  # not explicitly used by patch attacks
        if targeted:
            logger.warning("Patch attack generation may ignore 'targeted' set to True")
        attack_config["kwargs"] = kwargs

    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])
    attack = attack_fn(classifier, **attack_config["kwargs"])
    if not isinstance(attack, Attack):
        logger.warning(
            f"attack {attack} is not an instance of {Attack}."
            " Ensure that it implements ART `generate` API."
        )
    if attack_config.get("type") == "patch":
        attack_config["kwargs"] = original_kwargs
        return patch.AttackWrapper(attack, apply_patch_args, apply_patch_kwargs)
    return attack


def load_adversarial_dataset(config, num_batches=None, **kwargs):
    if config.get("type") != "preloaded":
        raise ValueError(f"attack type must be 'preloaded', not {config.get('type')}")
    dataset_module = import_module(config["module"])
    dataset_fn = getattr(dataset_module, config["name"])
    dataset_kwargs = config["kwargs"]
    dataset_kwargs.update(kwargs)
    if "description" in dataset_kwargs:
        dataset_kwargs.pop("description")
    dataset = dataset_fn(**dataset_kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    if config.get("check_run"):
        return EvalGenerator(dataset, num_eval_batches=1)
    if num_batches:
        return EvalGenerator(dataset, num_eval_batches=num_batches)
    return dataset


def _check_defense_api(defense, defense_baseclass):
    if not isinstance(defense, defense_baseclass):
        raise ValueError(
            f"defense {defense} does not extend type {type(defense_baseclass)}"
        )


def load_defense_wrapper(defense_config, classifier):
    defense_type = defense_config["type"]
    if defense_type == "Transformer":
        raise NotImplementedError("Transformer API not yet implemented into scenarios")
    elif defense_type != "Trainer":
        raise ValueError(
            f"Wrapped defenses must be of type Trainer, found {defense_type}"
        )

    defense_module = import_module(defense_config["module"])
    defense_fn = getattr(defense_module, defense_config["name"])
    defense = defense_fn(classifier, **defense_config["kwargs"])
    _check_defense_api(defense, Trainer)

    return defense


def load_defense_internal(defense_config, classifier):
    defense = load(defense_config)

    defense_type = defense_config["type"]
    if defense_type == "Preprocessor":
        _check_defense_api(defense, Preprocessor)
        preprocessing_defences = classifier.get_params().get("preprocessing_defences")
        if preprocessing_defences:
            preprocessing_defences.append(defense)
        else:
            preprocessing_defences = [defense]
        classifier.set_params(preprocessing_defences=preprocessing_defences)

    elif defense_type == "Postprocessor":
        _check_defense_api(defense, Postprocessor)
        postprocessing_defences = classifier.get_params().get("postprocessing_defences")
        if postprocessing_defences:
            postprocessing_defences.append(defense)
        else:
            postprocessing_defences = [defense]
        classifier.set_params(postprocessing_defences=postprocessing_defences)
    else:
        raise ValueError(
            f"Internal defenses must be of either type [Preprocessor, Postprocessor], found {defense_type}"
        )

    return classifier


def load_label_targeter(config):
    scheme = config["scheme"].lower()
    if scheme == "fixed":
        value = config.get("value")
        return labels.FixedLabelTargeter(value)
    elif scheme == "string":
        value = config.get("value")
        return labels.FixedStringTargeter(value)
    elif scheme == "random":
        num_classes = config.get("num_classes")
        return labels.RandomLabelTargeter(num_classes)
    elif scheme == "round-robin":
        num_classes = config.get("num_classes")
        offset = config.get("offset", 1)
        return labels.RoundRobinTargeter(num_classes, offset)
    elif scheme == "manual":
        values = config.get("values")
        repeat = config.get("repeat", False)
        return labels.ManualTargeter(values, repeat)
    elif scheme == "identity":
        return labels.IdentityTargeter()
    elif scheme == "matched length":
        transcripts = config.get("transcripts")
        return labels.MatchedTranscriptLengthTargeter(transcripts)
    elif scheme == "object_detection_fixed":
        value = config.get("value")
        score = config.get("score", 1.0)
        return labels.ObjectDetectionFixedLabelTargeteer(value, score)
    else:
        raise ValueError(
            f'scheme {scheme} not in ("fixed", "random", "round-robin", "manual", "identity", "matched length")'
        )
