from importlib import import_module
import logging
logger = logging.getLogger(__name__)
from typing import *


import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


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
            image = Image.fromarray(np.uint8(image * 255))
            image = image.resize(size=(224, 224), resample=Image.BILINEAR)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, _ = model(image)
            h = h.detach().cpu().numpy()
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
            class_minority = 2
        if class_majority == 0 and class_minority == 1:
            majority_flags = ~majority_flags
    return majority_flags


def get_majority_subclass_binary_labels(model, non_filtered_list, filtered_list, device):
    """ Return a dictionary from classes to arrays designating 
        by 1 a majority member of that class, and by 0 a minority

        Requests 2 clusters from get_data_level_stats.
    """

    majority_labels, minority_labels, cluster_labels, _, _ = get_data_level_stats(model, non_filtered_list, filtered_list, device, n_clusters=2)

    # Note that majority label is a list of length N_classes.  
    # The majority label may not be the same in each class.
    # Similarly, cluster_labels is also a list of lists or arrays, one per class.

    binary_labels = {}

    for i, majority_label in enumerate(majority_labels):
        binary_labels[i] = np.array(cluster_labels[i]) == majority_label

    return binary_labels


def demo():
    """
    Example code demonstrating how the above two functions can be used along with the
    resnet18_bean_regularization model. DELETE when integration is complete.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a pretrained model
    from BEAN.utils.resnet18_bean_regularization import get_model

    print('Running on device:', device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    weights_path = "./models/BEAN-2-abs-dist-alpha-100.pt"
    print(weights_path)
    model_kwargs = {
        "data_means": [0.39382024, 0.4159701, 0.40887499],
        "data_stds": [0.18931773, 0.18901625, 0.19651154],
        "num_classes": 10,
    }
    
    model = get_model(weights_path, **model_kwargs)
    print("Instantiated model")

    # Get all training data in the same format as Armory
    # i.e., images with channel last and normalized to [0,1]
    import numpy as np
    from PIL import Image
    import glob

    root_dir = "resisc10/test"
    image_dirs = glob.glob(root_dir + "/*")
    image_dirs.sort()

    complete_data_list = []
    for c, d in enumerate(image_dirs):
        images = glob.glob(d + "/*.jpg")
        images.sort()
        for image in images:
            im = Image.open(image)
            im = np.array(im, dtype=np.float32)
            im = im / 255.0
            complete_data_list.append((im, c))
    print("Created complete data list")

    # Create a random list of data to act as data filtered by poisoning defense
    num_filtered_data = 100
    filtered_data_idx = np.random.choice(
        len(complete_data_list), size=num_filtered_data, replace=False
    )
    filtered_data_list = [complete_data_list[idx] for idx in filtered_data_idx]
    print("Created filtered data list")
    
    #remove the elements of filtered_data_list from complete_data_list
    complete_data_list = [i for j, i in enumerate(complete_data_list) if j not in filtered_data_idx]
    
    #print(len(complete_data_list), len(filtered_data_list))
    
    # Get statistics on the training data
    (
        sample_majority,
        sample_minority,
        sample_cluster_labels,
        sample_silhouette_values,
        sample_cluster_avg
    ) = get_data_level_stats(model, complete_data_list, filtered_data_list, device)
    print("Calculated statistics of filtered data")

# Essentially copied from armory.utils.config_loading for BEAN regularization.
def load_bean_model(model_config):
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
    """
    if not isinstance(model, Classifier):
        raise TypeError(f"{model} is not an instance of {Classifier}")
    """
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