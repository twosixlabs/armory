from importlib import import_module
import logging
logger = logging.getLogger(__name__)
from typing import *


import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


from armory.data.utils import maybe_download_weights_from_s3


def cluster_data(x: np.ndarray, 
                 random_seed: int = 42, 
                 n_clusters: int = 2) -> np.ndarray:
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_seed)
    cluster_labels = clusterer.fit_predict(x)
    return cluster_labels


def get_majority_flags(model: Callable,
                       x: Iterable, 
                       device: torch.device, 
                       n_clusters: int = 2) -> np.ndarray:
    activations, class_ids  = [], []
    for image, class_id in x:
        with torch.no_grad():
            image = convert_np_image_to_tensor(image, device)
            h = get_hidden_representation(image, model)
            activations.append(h)
            class_ids.append(class_id)
    activations = np.concatenate(activations) 
    class_ids = np.array(class_ids, dtype=np.int64)
    majority_flags = np.zeros_like(class_ids.flatten(), dtype=np.bool_)
    for class_id in set(class_ids):
        activations_id = activations[class_ids == class_id]
        cluster_labels_id = cluster_data(activations_id, n_clusters=n_clusters)
        majority_flags[class_ids == class_id] = cluster_labels_id.astype(np.bool_)
        counts = np.bincount(cluster_labels_id, minlength=2)
        class_majority = np.argmax(counts)
        class_minority = np.argmin(counts)
        if class_majority == class_minority:
            class_majority = 1
            class_minority = 0
        if class_majority == 0 and class_minority == 1:
            majority_flags = ~majority_flags
    return majority_flags


def convert_np_image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(np.uint8(image * 255))
    image = image.resize(size=(224, 224), resample=Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = torch.tensor(image).to(device)
    return image


def get_hidden_representation(image: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    hidden_representation, _ = model(image)
    hidden_representation = hidden_representation.detach().cpu().numpy()
    return hidden_representation


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
    model = model_fn(weights_path)
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