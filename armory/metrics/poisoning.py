from importlib import import_module
import copy

import numpy as np
from PIL import Image
import torch

from armory.data.utils import maybe_download_weights_from_s3
from armory.logs import log
from armory.metrics.statistical import class_bias, class_majority_mask


# An armory user may request one of these models under 'adhoc'/'explanatory_model'
EXPLANATORY_MODEL_CONFIGS = explanatory_model_configs = {
    "cifar10_silhouette_model": {
        "model_kwargs": {
            "data_means": [0.4914, 0.4822, 0.4465],
            "data_stds": [0.2471, 0.2435, 0.2616],
            "num_classes": 10,
        },
        "module": "armory.baseline_models.pytorch.resnet18_bean_regularization",
        "name": "get_model",
        "weights_file": "cifar10_explanatory_model_resnet18_bean.pt",
    },
    "gtsrb_silhouette_model": {
        "model_kwargs": {},
        "module": "armory.baseline_models.pytorch.micronnet_gtsrb_bean_regularization",
        "name": "get_model",
        "resize_image": False,
        "weights_file": "gtsrb_explanatory_model_micronnet_bean.pt",
    },
    "resisc10_silhouette_model": {
        "model_kwargs": {
            "data_means": [0.39382024, 0.4159701, 0.40887499],
            "data_stds": [0.18931773, 0.18901625, 0.19651154],
            "num_classes": 10,
        },
        "module": "armory.baseline_models.pytorch.resnet18_bean_regularization",
        "name": "get_model",
        "weights_file": "resisc10_explanatory_model_resnet18_bean.pt",
    },
}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ExplanatoryModel:
    def __init__(
        self,
        explanatory_model,
        resize_image=True,
        size=(224, 224),
        resample=Image.BILINEAR,
        device=DEVICE,
    ):
        if not callable(explanatory_model):
            raise ValueError(f"explanatory_model {explanatory_model} is not callable")
        self.explanatory_model = explanatory_model
        self.resize_image = bool(resize_image)
        self.size = size
        self.resample = resample
        self.device = device

    @classmethod
    def from_config(cls, model_config, **kwargs):
        if isinstance(model_config, str):
            if model_config not in EXPLANATORY_MODEL_CONFIGS:
                raise ValueError(
                    f"model_config {model_config}, if a str, must be in {EXPLANATORY_MODEL_CONFIGS.keys()}"
                )
            model_config = EXPLANATORY_MODEL_CONFIGS[model_config]
        if not isinstance(model_config, dict):
            raise ValueError(
                f"model_config {model_config} must be a str or dict, not {type(model_config)}"
            )
        model_config = copy.copy(model_config)
        model_config.update(kwargs)  # override config with kwargs
        keys = ("module", "name", "weights_file")
        for k in keys:
            if k not in model_config:
                raise ValueError(f"config key {k} is required")
        module, name, weights_file = (model_config.pop(k) for k in keys)
        model_kwargs = model_config.pop("model_kwargs", {})

        weights_path = maybe_download_weights_from_s3(
            weights_file, auto_expand_tars=True
        )
        model_module = import_module(module)
        model_fn = getattr(model_module, name)
        explanatory_model = model_fn(weights_path, **model_kwargs)

        return cls(explanatory_model, **model_config)

    def get_activations(self, x):
        """
        Return array of activations from input batch x
        """
        activations = []
        with torch.no_grad():
            x = self.preprocess(x)
            activation, _ = self.explanatory_model(x)
            activations.append(activation.detach().cpu().numpy())
        return np.concatenate(activations)

    @staticmethod
    def _preprocess(
        x, resize_image=True, size=(224, 224), resample=Image.BILINEAR, device=DEVICE
    ):
        if isinstance(x.dtype, np.floating):
            if x.min() < 0.0 or x.max() > 1.0:
                raise ValueError("Floating input not bound to [0.0, 1.0] range")

            if resize_image:
                x = np.round(x * 255).astype(np.uint8)
            elif x.dtype != np.float32:
                x = x.astype(np.float32)
        elif x.dtype == np.uint8:
            if not resize_image:
                x = x.astype(np.float32) / 255
        else:
            raise ValueError(
                f"Input must be of type np.uint8 or floating, not {x.dtype}"
            )
        if resize_image:
            images = []
            for i in range(len(x)):
                image = Image.fromarray(x[i])
                image = image.resize(size=size, resample=resample)
                images.append(np.array(image, dtype=np.float32))
            x = np.stack(images) / 255

        return torch.tensor(x).to(device)

    def preprocess(self, x):
        """
        Preprocess a batch of images
        """
        return type(self)._preprocess(
            x,
            self.resize_image,
            self.size,
            resample=self.resample,
            device=self.device,
        )


class FairnessMetrics:
    """
    This class will manage the computation of fairness metrics for the poisoning scenario.
    """

    def __init__(self, poisoning_config):
        """
        poisoning_config: the adhoc section of the config
        """
        self.explanatory_model = ExplanatoryModel.from_config(
            poisoning_config.get("explanatory_model")
        )

    def add_cluster_metrics(
        self,
        x_poison,
        y_poison,
        poison_index,
        predicted_clean_indices,
        test_x,
        test_y,
        train_set_class_labels,
        test_set_class_labels,
        test_y_pred,
        is_filtering_defense,
        hub,
    ):
        """
        Compute two metrics that compare 2 binary distributions:
            1 (Model Bias) compares the classification accuracy in a binary split of each class.
            2 (Filter Bias) compares the filtering rate on the same binary splits.
        Currently, we compute both SPD and chi^2 on the contingency tables of the distributions.

        Record and log results.

        x_poison: the poisoned training dataset
        y_poison: the labels of the poisoned dataset
        poison_index: the indices of the poisoned samples in x_poison and y_poison
        predicted_clean_indices: the indices of the samples that the filter believes to be unpoisoned
        test_x: x values of the test set
        test_y: y values of the test set
        train_set_class_labels: class labels in the training set
        test_set_class_labels: class labels in the test set
        test_y_pred: predictions of the primary model on test_x
        is_filtering_defense: whether the filtering metric(s) should be computed
        hub: hub to log to
        """
        # get majority ceilings on unpoisoned part of train set
        poisoned_mask = np.zeros_like(y_poison, dtype=bool)
        poisoned_mask[poison_index.astype(np.int64)] = True
        x_unpoisoned = x_poison[~poisoned_mask]
        y_unpoisoned = y_poison[~poisoned_mask]

        activations = self.explanatory_model.get_activations(x_unpoisoned)
        majority_mask_unpoisoned, majority_ceilings = class_majority_mask(
            activations,
            y_unpoisoned,  # TODO: check
        )

        # Metric 1 General model bias
        # Compares rate of correct predictions between binary clusters of each class
        test_set_preds = test_y_pred.argmax(axis=1)

        correct_prediction_mask_test_set = test_y == test_set_preds
        activations = self.explanatory_model.get_activations(test_x)
        majority_mask_test_set, _ = class_majority_mask(
            activations,
            test_y,  # TODO: check
            majority_ceilings=majority_ceilings,  # use ceilings computed from train set
        )

        chi2_spd = class_bias(
            test_y,
            majority_mask_test_set,
            correct_prediction_mask_test_set,
            test_set_class_labels,
        )
        for class_id in test_set_class_labels:
            chi2, spd = chi2_spd[class_id]
            hub.record(f"model_bias_chi^2_p_value_{str(class_id).zfill(2)}", chi2)
            hub.record(f"model_bias_spd_{str(class_id).zfill(2)}", spd)
            log.info(
                f"Model Subclass Bias for Class {str(class_id).zfill(2)}: chi^2 p-value = {chi2:.4f}"
            )
            log.info(
                f"Model Subclass Bias for Class {str(class_id).zfill(2)}: SPD = {spd:.4f}"
            )

        # Metric 2 Filter bias (only if filtering defense)
        # Compares rate of filtering between binary clusters of each class
        if is_filtering_defense:
            kept_mask = np.zeros_like(y_poison, dtype=bool)
            kept_mask[predicted_clean_indices] = True
            kept_mask_unpoisoned = kept_mask[~poisoned_mask]

            chi2_spd = class_bias(
                y_unpoisoned,
                majority_mask_unpoisoned,
                kept_mask_unpoisoned,
                train_set_class_labels,
            )

            for class_id in train_set_class_labels:
                chi2, spd = chi2_spd[class_id]
                hub.record(f"filter_bias_chi^2_p_value_{str(class_id).zfill(2)}", chi2)
                hub.record(f"filter_bias_spd_{str(class_id).zfill(2)}", spd)
                if chi2 is None or spd is None:
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: not computed--the entire class was poisoned"
                    )
                else:
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: chi^2 p-value = {chi2:.4f}"
                    )
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: SPD = {spd:.4f}"
                    )
