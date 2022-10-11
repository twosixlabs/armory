from importlib import import_module
from typing import NamedTuple, Iterable, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import torch

from armory.data.utils import maybe_download_weights_from_s3
from armory.logs import log
from armory.instrument import Meter
from armory.utils.metrics import get_supported_metric, make_contingency_tables


class SilhouetteData(NamedTuple):
    n_clusters: int
    cluster_labels: np.ndarray
    silhouette_scores: np.ndarray


def get_majority_mask(
    explanatory_model: torch.nn.Module,
    data: Iterable,
    device: torch.device,
    resize_image: bool,
    majority_ceilings: Dict[int, float] = {},
    range_n_clusters: List[int] = [2],
    random_seed: int = 42,
) -> Tuple[np.ndarray, Dict[int, float]]:
    majority_mask = np.empty(len(data), dtype=np.bool_)
    activations, class_ids = _get_activations_w_class_ids(
        explanatory_model, data, device, resize_image
    )
    for class_id in np.unique(class_ids):
        majority_ceiling_id = majority_ceilings.get(class_id, None)
        mask_id = class_ids == class_id
        activations_id = activations[mask_id]
        silhouette_analysis_id = _get_silhouette_analysis(
            activations_id, range_n_clusters, random_seed
        )
        best_n_clusters_id = _get_best_n_clusters(silhouette_analysis_id)
        best_silhouette_data_id = silhouette_analysis_id[best_n_clusters_id]
        majority_mask_id, majority_ceiling_id = _get_majority_mask(
            best_silhouette_data_id, majority_ceiling_id
        )
        majority_mask[mask_id] = majority_mask_id
        majority_ceilings[class_id] = majority_ceiling_id
    return majority_mask, majority_ceilings


def _get_activations_w_class_ids(
    explanatory_model: torch.nn.Module,
    data: Iterable,
    device: torch.device,
    resize_image: bool,
) -> Tuple[np.ndarray]:
    activations, class_ids = [], []
    for image, class_id in data:
        with torch.no_grad():
            image = _convert_np_image_to_tensor(image, device, resize_image)
            activation = _get_activation(explanatory_model, image)
            activations.append(activation)
            class_ids.append(class_id)
    activations = np.concatenate(activations)
    class_ids = np.array(class_ids, dtype=np.int64)
    return activations, class_ids


def _convert_np_image_to_tensor(
    image: np.ndarray, device: torch.device, resize_image: bool
) -> torch.Tensor:
    image = Image.fromarray(np.uint8(image * 255))
    if resize_image:
        image = image.resize(size=(224, 224), resample=Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = torch.tensor(image).to(device)
    return image


def _get_activation(
    explanatory_model: torch.nn.Module, image: torch.Tensor
) -> np.ndarray:
    activation, _ = explanatory_model(image)
    activation = activation.detach().cpu().numpy()
    return activation


def _get_silhouette_analysis(
    activations: np.ndarray, range_n_clusters: List[int], random_seed: int
) -> Dict[int, SilhouetteData]:
    silhouette_analysis = {}
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_seed)
        cluster_labels = clusterer.fit_predict(activations)
        silhouette_scores = silhouette_samples(activations, cluster_labels)
        silhouette_data = SilhouetteData(n_clusters, cluster_labels, silhouette_scores)
        silhouette_analysis[n_clusters] = silhouette_data
    return silhouette_analysis


def _get_best_n_clusters(silhouette_analysis: Dict[int, SilhouetteData]) -> int:
    best_n_clusters = max(
        list(silhouette_analysis.keys()),
        key=lambda n_clusters: silhouette_analysis[n_clusters].silhouette_scores.mean(),
    )
    return best_n_clusters


def _get_majority_mask(
    silhouette_data: SilhouetteData, majority_ceiling: Optional[float]
) -> Tuple[np.ndarray, float]:
    if majority_ceiling is None:
        majority_ceiling = _get_mean_silhouette_score(silhouette_data)
    majority_mask = (0 <= silhouette_data.silhouette_scores) & (
        silhouette_data.silhouette_scores <= majority_ceiling
    )
    return majority_mask, majority_ceiling


def _get_mean_silhouette_score(silhouette_data: SilhouetteData) -> float:
    mean_silhouette_score = silhouette_data.silhouette_scores.mean()
    return mean_silhouette_score


# Essentially copied from armory.utils.config_loading for BEAN regularization.
def load_explanatory_model(model_config):
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
    try:
        model = model_fn(weights_path)
    except TypeError:
        model = model_fn(weights_path, model_config["model_kwargs"])
    if not weights_file and not model_config["fit"]:
        log.warning(
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


gtsrb_silhouette_clustering_config = {
    "fit": False,
    "fit_kwargs": {},
    "model_kwargs": {},
    "module": "armory.baseline_models.pytorch.micronnet_gtsrb_bean_regularization",
    "name": "get_model",
    "resize_image": False,
    "weights_file": "gtsrb_explanatory_model_micronnet_bean.pt",
    "wrapper_kwargs": {},
}

resisc10_silhouette_clustering_config = {
    "fit": False,
    "fit_kwargs": {},
    "model_kwargs": {
        "data_means": [0.39382024, 0.4159701, 0.40887499],
        "data_stds": [0.18931773, 0.18901625, 0.19651154],
        "num_classes": 10,
    },
    "module": "armory.baseline_models.pytorch.resnet18_bean_regularization",
    "name": "get_model",
    "weights_file": "resisc10_explanatory_model_resnet18_bean.pt",
    "wrapper_kwargs": {},
}

cifar10_silhouette_clustering_config = {
    "fit": False,
    "fit_kwargs": {},
    "model_kwargs": {
        "data_means": [0.4914, 0.4822, 0.4465],
        "data_stds": [0.2471, 0.2435, 0.2616],
        "num_classes": 10,
    },
    "module": "armory.baseline_models.pytorch.resnet18_bean_regularization",
    "name": "get_model",
    "weights_file": "cifar10_explanatory_model_resnet18_bean.pt",
    "wrapper_kwargs": {},
}


# An armory user will request one of these explanatory models under 'adhoc'/'explanatory_model'
explanatory_model_configs = {
    "gtsrb_silhouette_model": gtsrb_silhouette_clustering_config,
    "resisc10_silhouette_model": resisc10_silhouette_clustering_config,
    "cifar10_silhouette_model": cifar10_silhouette_clustering_config,
}


class FairnessMetrics:
    """This class will manage the computation of fairness metrics for the poisoning scenario."""

    def __init__(self, poisoning_config, is_filtering_defense, scenario):
        """poisoning_config: the adhoc section of the config
        is_filtering_defense: Boolean used to indicate whether the filtering metric(s) should be computed
        scenario: A reference to the scenario object which instantiates this
        """
        # self.metric_config = metric_config
        self.is_filtering_defense = is_filtering_defense
        self.DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.scenario = scenario

        explanatory_model_name = poisoning_config.get("explanatory_model", None)
        if explanatory_model_name not in explanatory_model_configs.keys():
            raise ValueError(
                f"Config should specify model for fairness metrics.  Set adhoc/explanatory_model to one of {list(explanatory_model_configs.keys())}, "
                + "or set adhoc/compute_fairness_metrics to false if these metrics are not desired."
            )
        explanatory_model_config = explanatory_model_configs[explanatory_model_name]
        explanatory_model, _ = load_explanatory_model(explanatory_model_config)
        self.explanatory_model = explanatory_model
        self.explanatory_model_resize_image = explanatory_model_config.get(
            "resize_image", True
        )

    def add_filter_perplexity(
        self,
        result_name="filter_perplexity",
        y_clean_name="scenario.y_clean",
        poison_index_name="scenario.poison_index",
        predicted_clean_indices="scenario.is_dirty_mask",
    ):
        """Compute filter perplexity, add it to the results dict in the calling scenario, and return data for logging

        y_clean: the labels for the clean dataset
        poison_index: the indices of the poisoned samples
        predicted_clean_indices: the indices of the samples that the filter believes to be unpoisoned
        """
        self.scenario.hub.connect_meter(
            Meter(
                f"input_to_{result_name}",
                get_supported_metric("filter_perplexity_fps_benign"),
                y_clean_name,
                poison_index_name,
                predicted_clean_indices,
                final=np.mean,
                final_name=result_name,
                record_final_only=True,
            )
        )

    def add_cluster_metrics(
        self,
        x_poison,
        y_poison,
        poison_index,
        predicted_clean_indices,
        test_dataset,
        train_set_class_labels,
        test_set_class_labels,
    ):
        """Compute two metrics based on comparing two binary distributions.
        Metric 1 (Model Bias) compares the classification accuracy in a binary split of each class.
        Metric 2 (Filter Bias) compares the filtering rate on the same binary splits.
        This comparison can be made in a variety of arbitrary ways.  Currently, we compute both SPD and chi^2 on
        the contingency tables of the distributions.

        Adds results to results dict of calling scenario, and returns the data for logging.

        x_poison: the poisoned training dataset
        y_poison: the labels of the poisoned dataset
        poison_index: the indices of the poisoned samples in x_poison and y_poison
        predicted_clean_indices: the indices of the samples that the filter believes to be unpoisoned
        test_dataset: the test dataset
        """
        # get majority ceilings on unpoisoned part of train set
        poisoned_mask = np.zeros_like(y_poison, dtype=np.bool_)
        poisoned_mask[poison_index.astype(np.int64)] = True
        x_unpoisoned = x_poison[~poisoned_mask]
        y_unpoisoned = y_poison[~poisoned_mask]
        majority_mask_unpoisoned, majority_ceilings = get_majority_mask(
            explanatory_model=self.explanatory_model,
            data=list(zip(x_unpoisoned, y_unpoisoned)),
            device=self.DEVICE,
            resize_image=self.explanatory_model_resize_image,
        )

        # Metric 1 General model bias
        # Compares rate of correct predictions between binary clusters of each class
        test_x, test_y = (np.concatenate(z, axis=0) for z in zip(*list(test_dataset)))
        test_set_preds = self.scenario.model.predict(
            test_x, **self.scenario.predict_kwargs
        ).argmax(1)
        correct_prediction_mask_test_set = test_y == test_set_preds
        majority_mask_test_set, majority_ceilings = get_majority_mask(
            explanatory_model=self.explanatory_model,
            data=list(zip(test_x, test_y)),
            majority_ceilings=majority_ceilings,  # use ceilings computed from train set
            device=self.DEVICE,
            resize_image=self.explanatory_model_resize_image,
        )

        majority_x_correct_prediction_tables = make_contingency_tables(
            test_y, majority_mask_test_set, correct_prediction_mask_test_set
        )

        chi2_metric = get_supported_metric("poison_chi2_p_value")
        spd_metric = get_supported_metric("poison_spd")
        for class_id in test_set_class_labels:
            majority_x = majority_x_correct_prediction_tables[class_id]
            chi2 = np.mean(chi2_metric(majority_x))
            spd = np.mean(spd_metric(majority_x))
            self.scenario.hub.record(
                f"model_bias_chi^2_p_value_{str(class_id).zfill(2)}", chi2
            )
            self.scenario.hub.record(f"model_bias_spd_{str(class_id).zfill(2)}", spd)
            log.info(
                f"Model Subclass Bias for Class {str(class_id).zfill(2)}: chi^2 p-value = {chi2:.4f}"
            )
            log.info(
                f"Model Subclass Bias for Class {str(class_id).zfill(2)}: SPD = {spd:.4f}"
            )

        # Metric 2 Filter bias (only if filtering defense)
        # Compares rate of filtering between binary clusters of each class
        if self.is_filtering_defense:
            kept_mask = np.zeros_like(y_poison, dtype=np.bool_)
            kept_mask[predicted_clean_indices] = True
            kept_mask_unpoisoned = kept_mask[~poisoned_mask]

            majority_x_passed_filter_tables = make_contingency_tables(
                y_unpoisoned, majority_mask_unpoisoned, kept_mask_unpoisoned
            )

            for class_id in train_set_class_labels:
                try:
                    majority_x = majority_x_passed_filter_tables[class_id]
                    chi2 = np.mean(chi2_metric(majority_x))
                    spd = np.mean(spd_metric(majority_x))
                    self.scenario.hub.record(
                        f"filter_bias_chi^2_p_value_{str(class_id).zfill(2)}", chi2
                    )
                    self.scenario.hub.record(
                        f"filter_bias_spd_{str(class_id).zfill(2)}", spd
                    )
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: chi^2 p-value = {chi2:.4f}"
                    )
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: SPD = {spd:.4f}"
                    )
                except KeyError:
                    self.scenario.hub.record(
                        f"filter_bias_chi^2_p_value_{str(class_id).zfill(2)}", None
                    )
                    self.scenario.hub.record(
                        f"filter_bias_spd_{str(class_id).zfill(2)}", None
                    )
                    log.info(
                        f"Filter Subclass Bias for Class {str(class_id).zfill(2)}: not computed--the entire class was poisoned"
                    )
