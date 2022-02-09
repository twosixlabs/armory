from importlib import import_module
import logging
logger = logging.getLogger(__name__)
from typing import *


import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


from armory.data.utils import maybe_download_weights_from_s3


class SilhouetteData(NamedTuple):
    n_clusters: int
    cluster_labels: np.ndarray
    silhouette_scores: np.ndarray


def get_majority_mask(explanatory_model: torch.nn.Module,
                      data: Iterable,
                      device: torch.device,
                      resize_image: bool,
                      range_n_clusters: List[int] = [2],
                      random_seed: int = 42) -> np.ndarray:
    majority_mask = np.empty(len(data), dtype=np.bool_)
    activations, class_ids = _get_activations_w_class_ids(explanatory_model, 
                                                          data, 
                                                          device, 
                                                          resize_image)
    for class_id in np.unique(class_ids):
        mask_id = (class_ids == class_id)
        activations_id = activations[mask_id]
        silhouette_analysis_id = _get_silhouette_analysis(activations_id, 
                                                          range_n_clusters, 
                                                          random_seed)
        best_n_clusters_id = _get_best_n_clusters(silhouette_analysis_id)
        best_silhouette_data_id = silhouette_analysis_id[best_n_clusters_id]
        majority_mask_id = _get_majority_mask(best_silhouette_data_id)
        majority_mask[mask_id] = majority_mask_id
    return majority_mask


def _get_activations_w_class_ids(explanatory_model: torch.nn.Module,
                                 data: Iterable,
                                 device: torch.device,
                                 resize_image: bool) -> Tuple[np.ndarray]:
    activations, class_ids  = [], []
    for image, class_id in data:
        with torch.no_grad():
            image = _convert_np_image_to_tensor(image, device, resize_image)
            activation = _get_activation(explanatory_model, image)
            activations.append(activation)
            class_ids.append(class_id)
    activations = np.concatenate(activations) 
    class_ids = np.array(class_ids, dtype=np.int64)
    return activations, class_ids


def _convert_np_image_to_tensor(image: np.ndarray, 
                                device: torch.device, 
                                resize_image: bool) -> torch.Tensor:
    image = Image.fromarray(np.uint8(image * 255))
    if resize_image:
        image = image.resize(size=(224, 224), resample=Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = torch.tensor(image).to(device)
    return image


def _get_activation(explanatory_model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    activation, _ = explanatory_model(image)
    activation = activation.detach().cpu().numpy()
    return activation


def _get_silhouette_analysis(activations: np.ndarray, 
                             range_n_clusters: List[int],
                             random_seed: int) -> Dict[int, SilhouetteData]:
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
        key=lambda n_clusters: silhouette_analysis[n_clusters].silhouette_scores.mean()
    )
    return best_n_clusters


def _get_majority_mask(silhouette_data: SilhouetteData) -> np.ndarray:
    mean_silhouette_score = _get_mean_silhouette_score(silhouette_data)
    majority_mask = ((0 <= silhouette_data.silhouette_scores) & (silhouette_data.silhouette_scores <= mean_silhouette_score))
    return majority_mask


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
    except TypeError as e:
        model = model_fn(weights_path, model_config["model_kwargs"])
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