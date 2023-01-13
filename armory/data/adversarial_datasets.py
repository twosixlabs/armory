"""
Adversarial datasets
"""

from typing import Callable

import numpy as np
import tensorflow as tf

from armory.data import datasets
from armory.data.adversarial.apricot_metadata import ADV_PATCH_MAGIC_NUMBER_LABEL_ID

imagenet_adversarial_context = datasets.ImageContext(x_shape=(224, 224, 3))
librispeech_adversarial_context = datasets.AudioContext(
    x_shape=(None,), sample_rate=16000
)
resisc45_adversarial_context = datasets.ImageContext(x_shape=(224, 224, 3))
ucf101_adversarial_context = datasets.ImageContext(x_shape=(None, 112, 112, 3))
apricot_adversarial_context = datasets.ImageContext(x_shape=(None, None, 3))
dapricot_adversarial_context = datasets.ImageContext(x_shape=(3, None, None, 3))
carla_obj_det_dev_single_modal_context = datasets.ImageContext(x_shape=(960, 1280, 3))
carla_obj_det_dev_multimodal_context = datasets.ImageContext(x_shape=(960, 1280, 6))
carla_video_tracking_context = datasets.VideoContext(
    x_shape=(None, 960, 1280, 3), frame_rate=10
)
carla_obj_det_test_single_modal_context = datasets.ImageContext(x_shape=(960, 1280, 3))
carla_obj_det_test_multimodal_context = datasets.ImageContext(x_shape=(960, 1280, 6))


def imagenet_adversarial_canonical_preprocessing(batch):
    return datasets.canonical_image_preprocess(imagenet_adversarial_context, batch)


def librispeech_adversarial_canonical_preprocessing(batch):
    return datasets.canonical_audio_preprocess(librispeech_adversarial_context, batch)


def resisc45_adversarial_canonical_preprocessing(batch):
    return datasets.canonical_image_preprocess(resisc45_adversarial_context, batch)


def ucf101_adversarial_canonical_preprocessing(batch):
    return datasets.canonical_image_preprocess(ucf101_adversarial_context, batch)


def apricot_canonical_preprocessing(batch):
    return datasets.canonical_variable_image_preprocess(
        apricot_adversarial_context, batch
    )


def dapricot_canonical_preprocessing(batch):
    # DAPRICOT raw images are rotated by 90 deg and color channels are BGR, so the
    # following line corrects for this
    batch_rotated_rgb = np.transpose(batch, (0, 1, 3, 2, 4))[:, :, :, ::-1, :]
    return datasets.canonical_variable_image_preprocess(
        dapricot_adversarial_context, batch_rotated_rgb
    )


def carla_video_tracking_canonical_preprocessing(batch):
    return datasets.canonical_variable_image_preprocess(
        carla_video_tracking_context, batch
    )


def imagenet_adversarial(
    split: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = imagenet_adversarial_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial",
    targeted: bool = False,
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    """
    ILSVRC12 adversarial image dataset for ResNet50

    ProjectedGradientDescent
        Iterations = 10
        Max perturbation epsilon = 8
        Attack step size = 2
        Targeted = True
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    if adversarial_key != "adversarial":
        raise ValueError(f"{adversarial_key} != 'adversarial'")
    if targeted:
        raise ValueError(f"{adversarial_key} is not a targeted attack")

    return datasets._generator_from_tfds(
        "imagenet_adversarial:1.1.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: ((x[clean_key], x[adversarial_key]), y),
        context=imagenet_adversarial_context,
        **kwargs,
    )


def librispeech_adversarial(
    split: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = librispeech_adversarial_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_perturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    """
    Adversarial dataset based on Librispeech-dev-clean including clean,
    Universal Perturbation using PGD, and PGD.

    split - one of ("adversarial")

    returns:
        Generator
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_perturbation", "adversarial_univperturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        raise ValueError(f"{adversarial_key} is not a targeted attack")

    return datasets._generator_from_tfds(
        "librispeech_adversarial:1.1.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("audio", "label"),
        variable_length=bool(batch_size > 1),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: ((x[clean_key], x[adversarial_key]), y),
        context=librispeech_adversarial_context,
        **kwargs,
    )


def resisc45_adversarial_224x224(
    split: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = resisc45_adversarial_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_univperturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    """
    resisc45 Adversarial Dataset of size (224, 224, 3),
    including clean, adversarial universal perturbation, and
    adversarial patched
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_univpatch", "adversarial_univperturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        if adversarial_key == "adversarial_univperturbation":
            raise ValueError("adversarial_univperturbation is not a targeted attack")

        def lambda_map(x, y):
            return (
                (x[clean_key], x[adversarial_key]),
                (y[clean_key], y[adversarial_key]),
            )

    else:

        def lambda_map(x, y):
            return (x[clean_key], x[adversarial_key]), y[clean_key]

    return datasets._generator_from_tfds(
        "resisc45_densenet121_univpatch_and_univperturbation_adversarial224x224:1.0.2",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("images", "labels"),
        variable_length=False,
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda_map,
        context=resisc45_adversarial_context,
        **kwargs,
    )


def ucf101_adversarial_112x112(
    split: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = ucf101_adversarial_canonical_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = "clean",
    adversarial_key: str = "adversarial_perturbation",
    targeted: bool = False,
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    """
    UCF 101 Adversarial Dataset of size (112, 112, 3),
    including clean, adversarial perturbed, and
    adversarial patched

    DataGenerator returns batches of ((x_clean, x_adversarial), y)
    """
    if clean_key != "clean":
        raise ValueError(f"{clean_key} != 'clean'")
    adversarial_keys = ("adversarial_patch", "adversarial_perturbation")
    if adversarial_key not in adversarial_keys:
        raise ValueError(f"{adversarial_key} not in {adversarial_keys}")
    if targeted:
        if adversarial_key == "adversarial_perturbation":
            raise ValueError("adversarial_perturbation is not a targeted attack")

        def lambda_map(x, y):
            return (
                (x[clean_key], x[adversarial_key]),
                (y[clean_key], y[adversarial_key]),
            )

    else:

        def lambda_map(x, y):
            return (x[clean_key], x[adversarial_key]), y[clean_key]

    return datasets._generator_from_tfds(
        "ucf101_mars_perturbation_and_patch_adversarial112x112:1.1.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("videos", "labels"),
        variable_length=bool(batch_size > 1),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda_map,
        context=ucf101_adversarial_context,
        **kwargs,
    )


def gtsrb_poison(
    split: str = "poison",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = None,
    cache_dataset: bool = True,
    framework: str = "numpy",
    clean_key: str = None,
    adversarial_key: str = None,
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    """
    German traffic sign poison dataset of size (48, 48, 3),
    including only poisoned data

    DataGenerator returns batches of (x_poison, y)
    """
    return datasets._generator_from_tfds(
        "gtsrb_bh_poison_micronnet:1.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", "label"),
        variable_length=bool(batch_size > 1),
        shuffle_files=False,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: (x, y),
        **kwargs,
    )


def apricot_label_preprocessing(x, y):
    """
    Convert labels to list of dicts. If batch_size > 1, this will already be the case.
    Decrement labels of non-patch objects by 1 to be 0-indexed
    """
    if isinstance(y, dict):
        y = [y]
    for y_dict in y:
        y_dict["labels"] -= y_dict["labels"] != ADV_PATCH_MAGIC_NUMBER_LABEL_ID
        y_dict["labels"] = y_dict["labels"].reshape((-1,))
    return y


def dapricot_label_preprocessing(x, y):
    """ """
    y_object, y_patch_metadata = y
    y_object_list = []
    y_patch_metadata_list = []
    # each example contains images from N cameras, i.e. N=3
    num_imgs_per_ex = np.array(y_object["id"].flat_values).size
    y_patch_metadata["gs_coords"] = np.array(
        y_patch_metadata["gs_coords"].flat_values
    ).reshape((num_imgs_per_ex, -1, 2))
    y_patch_metadata["shape"] = y_patch_metadata["shape"].reshape((num_imgs_per_ex,))
    y_patch_metadata["cc_scene"] = y_patch_metadata["cc_scene"][0]
    y_patch_metadata["cc_ground_truth"] = y_patch_metadata["cc_ground_truth"][0]
    for i in range(num_imgs_per_ex):
        y_object_img = {}
        for k, v in y_object.items():
            y_object_img[k] = np.expand_dims(
                np.array(y_object[k].flat_values[i]), axis=0
            )
        y_object_list.append(y_object_img)

        y_patch_metadata_img = {
            k: np.array(y_patch_metadata[k][i]) for k, v in y_patch_metadata.items()
        }
        y_patch_metadata_list.append(y_patch_metadata_img)

    return (y_object_list, y_patch_metadata_list)


def apricot_dev_adversarial(
    split: str = "frcnn+ssd+retinanet",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = apricot_canonical_preprocessing,
    label_preprocessing_fn: Callable = apricot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    if batch_size != 1:
        raise NotImplementedError("Currently working only with batch size = 1")
    if "class_ids" in kwargs:
        raise ValueError("Filtering by class is not supported for the APRICOT dataset")

    # The apricot dataset uses 12 as the label for adversarial patches, which may be used for
    # meaningful categories for other datasets. This method is applied as a lambda_map to convert
    #  this label from 12 to the ADV_PATCH_MAGIC_NUMBER_LABEL_ID -- we choose a negative integer
    #  for the latter since it is unlikely that such a number represents the ID of a class in
    # another dataset
    raw_adv_patch_category_id = 12

    def replace_magic_val(data, raw_val, transformed_val, sub_key):
        rhs = data[sub_key]
        data[sub_key] = tf.where(
            tf.equal(rhs, raw_val),
            tf.ones_like(rhs, dtype=tf.int64) * transformed_val,
            rhs,
        )
        return data

    if split == "adversarial":
        split = "frcnn+ssd+retinanet"

    return datasets._generator_from_tfds(
        "apricot_dev:1.0.2",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", "objects"),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: (
            x,
            replace_magic_val(
                y,
                raw_adv_patch_category_id,
                ADV_PATCH_MAGIC_NUMBER_LABEL_ID,
                "labels",
            ),
        ),
        context=apricot_adversarial_context,
        **kwargs,
    )


def apricot_test_adversarial(
    split: str = "adversarial",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = apricot_canonical_preprocessing,
    label_preprocessing_fn: Callable = apricot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
) -> datasets.ArmoryDataGenerator:
    if batch_size != 1:
        raise NotImplementedError("Currently working only with batch size = 1")
    if "class_ids" in kwargs:
        raise ValueError("Filtering by class is not supported for the APRICOT dataset")

    # The apricot dataset uses 12 as the label for adversarial patches, which may be used for
    # meaningful categories for other datasets. This method is applied as a lambda_map to convert
    #  this label from 12 to the ADV_PATCH_MAGIC_NUMBER_LABEL_ID -- we choose a negative integer
    #  for the latter since it is unlikely that such a number represents the ID of a class in
    # another dataset
    raw_adv_patch_category_id = 12

    if split == "adversarial":
        split = "frcnn+ssd+retinanet"

    def replace_magic_val(data, raw_val, transformed_val, sub_key):
        rhs = data[sub_key]
        data[sub_key] = tf.where(
            tf.equal(rhs, raw_val),
            tf.ones_like(rhs, dtype=tf.int64) * transformed_val,
            rhs,
        )
        return data

    return datasets._generator_from_tfds(
        "apricot_test:1.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", "objects"),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        lambda_map=lambda x, y: (
            x,
            replace_magic_val(
                y,
                raw_adv_patch_category_id,
                ADV_PATCH_MAGIC_NUMBER_LABEL_ID,
                "labels",
            ),
        ),
        context=apricot_adversarial_context,
        **kwargs,
    )


def dapricot_dev_adversarial(
    split: str = "large+medium+small",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = dapricot_canonical_preprocessing,
    label_preprocessing_fn: Callable = dapricot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    if batch_size != 1:
        raise ValueError("D-APRICOT batch size must be set to 1")

    if split == "adversarial":
        split = "small+medium+large"

    return datasets._generator_from_tfds(
        "dapricot_dev:1.0.1",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        context=dapricot_adversarial_context,
    )


def dapricot_test_adversarial(
    split: str = "large+medium+small",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = dapricot_canonical_preprocessing,
    label_preprocessing_fn: Callable = dapricot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
) -> datasets.ArmoryDataGenerator:
    if batch_size != 1:
        raise ValueError("D-APRICOT batch size must be set to 1")

    if split == "adversarial":
        split = "small+medium+large"

    return datasets._generator_from_tfds(
        "dapricot_test:1.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        shuffle_files=shuffle_files,
        cache_dataset=cache_dataset,
        framework=framework,
        context=dapricot_adversarial_context,
    )


def carla_obj_det_label_preprocessing(x, y):
    y_object, y_patch_metadata = y
    y_object = {k: np.squeeze(v, axis=0) for k, v in y_object.items()}
    y_patch_metadata = {k: np.squeeze(v, axis=0) for k, v in y_patch_metadata.items()}

    # convert TF format to PyTorch format of [x1, y1, x2, y2]
    height, width = x.shape[2:4]
    converted_boxes = y_object["boxes"][:, [1, 0, 3, 2]]
    converted_boxes *= [width, height, width, height]
    y_object["boxes"] = converted_boxes
    return (y_object, y_patch_metadata)


def carla_obj_det_dev_canonical_preprocessing(batch):
    if batch.shape[-1] == 6:
        context = carla_obj_det_dev_multimodal_context
    else:
        context = carla_obj_det_dev_single_modal_context
    return datasets.canonical_image_preprocess(context, batch)


def carla_obj_det_test_canonical_preprocessing(batch):
    if batch.shape[-1] == 6:
        context = carla_obj_det_test_multimodal_context
    else:
        context = carla_obj_det_test_single_modal_context
    return datasets.canonical_image_preprocess(context, batch)


def carla_obj_det_dev(
    split: str = "dev",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_obj_det_dev_canonical_preprocessing,
    label_preprocessing_fn=carla_obj_det_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
):
    """
    Dev set for CARLA object detection dataset, containing RGB and depth channels. The dev
    set also contains green screens for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_obj_det_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_dev batch size must be set to 1")

    modality = kwargs.pop("modality", "rgb")
    if modality not in ["rgb", "depth", "both"]:
        raise ValueError(
            'Unknown modality: {}.  Must be one of "rgb", "depth", or "both"'.format(
                modality
            )
        )

    def rgb_fn(batch):
        return batch[:, 0]

    def depth_fn(batch):
        return batch[:, 1]

    def both_fn(batch):
        return np.concatenate((batch[:, 0], batch[:, 1]), axis=-1)

    func_dict = {"rgb": rgb_fn, "depth": depth_fn, "both": both_fn}
    mode_split_fn = func_dict[modality]
    preprocessing_fn = datasets.preprocessing_chain(mode_split_fn, preprocessing_fn)

    context = (
        carla_obj_det_dev_multimodal_context
        if modality == "both"
        else carla_obj_det_dev_single_modal_context
    )

    return datasets._generator_from_tfds(
        "carla_obj_det_dev:2.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=context,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        **kwargs,
    )


def carla_over_obj_det_dev(
    split: str = "dev",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_obj_det_dev_canonical_preprocessing,
    label_preprocessing_fn=carla_obj_det_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
):
    """
    Dev set for CARLA object detection dataset, containing RGB and depth channels. The dev
    set also contains green screens for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_obj_det_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_dev batch size must be set to 1")

    modality = kwargs.pop("modality", "rgb")
    if modality not in ["rgb", "depth", "both"]:
        raise ValueError(
            'Unknown modality: {}.  Must be one of "rgb", "depth", or "both"'.format(
                modality
            )
        )

    def rgb_fn(batch):
        return batch[:, 0]

    def depth_fn(batch):
        return batch[:, 1]

    def both_fn(batch):
        return np.concatenate((batch[:, 0], batch[:, 1]), axis=-1)

    func_dict = {"rgb": rgb_fn, "depth": depth_fn, "both": both_fn}
    mode_split_fn = func_dict[modality]
    preprocessing_fn = datasets.preprocessing_chain(mode_split_fn, preprocessing_fn)

    context = (
        carla_obj_det_dev_multimodal_context
        if modality == "both"
        else carla_obj_det_dev_single_modal_context
    )

    return datasets._generator_from_tfds(
        "carla_over_obj_det_dev:1.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=context,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        **kwargs,
    )


def carla_over_obj_det_test(
    split: str = "test",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_obj_det_test_canonical_preprocessing,
    label_preprocessing_fn=carla_obj_det_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
):
    """
    Dev set for CARLA object detection dataset, containing RGB and depth channels. The test
    set also contains green screens for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_obj_det_test dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_test batch size must be set to 1")

    modality = kwargs.pop("modality", "rgb")
    if modality not in ["rgb", "depth", "both"]:
        raise ValueError(
            'Unknown modality: {}.  Must be one of "rgb", "depth", or "both"'.format(
                modality
            )
        )

    def rgb_fn(batch):
        return batch[:, 0]

    def depth_fn(batch):
        return batch[:, 1]

    def both_fn(batch):
        return np.concatenate((batch[:, 0], batch[:, 1]), axis=-1)

    func_dict = {"rgb": rgb_fn, "depth": depth_fn, "both": both_fn}
    mode_split_fn = func_dict[modality]
    preprocessing_fn = datasets.preprocessing_chain(mode_split_fn, preprocessing_fn)

    context = (
        carla_obj_det_test_multimodal_context
        if modality == "both"
        else carla_obj_det_test_single_modal_context
    )

    return datasets._generator_from_tfds(
        "carla_over_obj_det_test:1.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=context,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        **kwargs,
    )


def carla_obj_det_test(
    split: str = "test",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_obj_det_test_canonical_preprocessing,
    label_preprocessing_fn=carla_obj_det_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs,
):
    """
    Test set for CARLA object detection dataset, containing RGB and depth channels. The test
    set also contains green screens for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_obj_det_test dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_test batch size must be set to 1")

    modality = kwargs.pop("modality", "rgb")
    if modality not in ["rgb", "depth", "both"]:
        raise ValueError(
            'Unknown modality: {}.  Must be one of "rgb", "depth", or "both"'.format(
                modality
            )
        )

    def rgb_fn(batch):
        return batch[:, 0]

    def depth_fn(batch):
        return batch[:, 1]

    def both_fn(batch):
        return np.concatenate((batch[:, 0], batch[:, 1]), axis=-1)

    func_dict = {"rgb": rgb_fn, "depth": depth_fn, "both": both_fn}
    mode_split_fn = func_dict[modality]
    preprocessing_fn = datasets.preprocessing_chain(mode_split_fn, preprocessing_fn)

    context = (
        carla_obj_det_test_multimodal_context
        if modality == "both"
        else carla_obj_det_test_single_modal_context
    )

    return datasets._generator_from_tfds(
        "carla_obj_det_test:2.0.0",
        split=split,
        batch_size=batch_size,
        epochs=epochs,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=context,
        as_supervised=False,
        supervised_xy_keys=("image", ("objects", "patch_metadata")),
        **kwargs,
    )


class ClipVideoTrackingLabels:
    """
    Truncate labels for CARLA video tracking, when max_frames is set
    """

    def __init__(self, max_frames):
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames {max_frames} must be > 0")
        self.max_frames = max_frames

    def clip_boxes(self, boxes):
        return boxes[:, : self.max_frames, :]

    def clip_metadata(self, patch_metadata_dict):
        return {
            k: v[:, : self.max_frames, ::] for (k, v) in patch_metadata_dict.items()
        }

    def __call__(self, x, labels):
        boxes, patch_metadata_dict = labels
        return self.clip_boxes(boxes), self.clip_metadata(patch_metadata_dict)


class ClipMOTVideoTrackingLabels(ClipVideoTrackingLabels):
    """
    Truncate labels for CARLA multi-object video tracking, when max_frames is set
        Assumes zero indexing for frames
        Also assumes shape is (batch, num_detections, 9)
            The first value in the 9-dim vector is the frame_id
    """

    def clip_boxes(self, boxes):
        if len(boxes) == 1:
            boxes_out = np.expand_dims(boxes[boxes[:, :, 0] < self.max_frames], 0)
        else:
            boxes_out = np.empty(len(boxes), dtype=object)
            for i, b in enumerate(boxes):
                boxes_out[i] = b[b[:, 0] < self.max_frames]
        return boxes_out


def carla_video_tracking_label_preprocessing(x, y):
    box_labels, patch_metadata = y
    box_array = np.squeeze(box_labels, axis=0)
    box_labels = [{"boxes": box_array}]
    patch_metadata = {k: np.squeeze(v, axis=0) for k, v in patch_metadata.items()}
    return (box_labels, patch_metadata)


def carla_video_tracking_dev(
    split: str = "dev",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_video_tracking_canonical_preprocessing,
    label_preprocessing_fn=carla_video_tracking_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    max_frames: int = None,
    **kwargs,
):
    """
    Dev set for CARLA video tracking dataset, The dev set also contains green screens
    for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_video_tracking_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_dev batch size must be set to 1")

    if max_frames:
        clip = datasets.ClipFrames(max_frames)
        clip_labels = ClipVideoTrackingLabels(max_frames)
    else:
        clip = None
        clip_labels = None

    preprocessing_fn = datasets.preprocessing_chain(clip, preprocessing_fn)
    label_preprocessing_fn = datasets.label_preprocessing_chain(
        clip_labels, label_preprocessing_fn
    )

    return datasets._generator_from_tfds(
        "carla_video_tracking_dev:2.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=carla_video_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", ("bboxes", "patch_metadata")),
        **kwargs,
    )


def carla_video_tracking_test(
    split: str = "test",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_video_tracking_canonical_preprocessing,
    label_preprocessing_fn=carla_video_tracking_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    max_frames: int = None,
    **kwargs,
):
    """
    Test set for CARLA video tracking dataset, The test set also contains green screens
    for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_video_tracking_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_obj_det_dev batch size must be set to 1")

    if max_frames:
        clip = datasets.ClipFrames(max_frames)
        clip_labels = ClipVideoTrackingLabels(max_frames)
    else:
        clip = None
        clip_labels = None

    preprocessing_fn = datasets.preprocessing_chain(clip, preprocessing_fn)
    label_preprocessing_fn = datasets.label_preprocessing_chain(
        clip_labels, label_preprocessing_fn
    )

    return datasets._generator_from_tfds(
        "carla_video_tracking_test:2.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=carla_video_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", ("bboxes", "patch_metadata")),
        **kwargs,
    )


def carla_mot_label_preprocessing(x, y):
    annotations, patch_metadata = y
    patch_metadata = {k: np.squeeze(v, axis=0) for k, v in patch_metadata.items()}
    return (annotations, patch_metadata)


def mot_zero_index(x, y):
    annotations, patch_metadata = y
    if annotations.ndim == 2:
        annotations[:, 0] -= 1
    else:
        for annotation in annotations:
            annotation[:, 0] -= 1
    return (annotations, patch_metadata)


def mot_array_to_coco(batch):
    """
    Map from 3D array (batch_size x detections x 9) to extended coco format
        of dimension (batch_size x frames x detections_per_frame)

    NOTE: 'image_id' is given as the frame of a video, so is not unique
    """
    if batch.ndim == 2:
        not_batch = True
        batch = [batch]
    elif batch.ndim == 3:
        not_batch = False
    else:
        raise ValueError(f"batch.ndim {batch.ndim} is not in (2, 3)")

    output = np.empty(len(batch), dtype=object)
    for i, array in enumerate(batch):
        if not len(array):
            # no object detections
            output.append([])
            continue

        frames = []
        for detection in array:
            frames.append(
                {
                    # TODO: should image_id include video number as well?
                    "image_id": int(np.round(detection[0])),
                    "category_id": int(np.round(detection[7])),
                    "bbox": [float(x) for x in detection[2:6]],
                    "score": float(detection[6]),
                    # The following are extended fields
                    "object_id": int(
                        np.round(detection[1])
                    ),  # for a specific object across frames
                    "visibility": float(detection[8]),
                }
            )
        output[i] = frames

    if not_batch:
        output = output[0]

    return output


def mot_array_to_coco_label_preprocessing(x, y):
    """
    x is given in the label preprocessing pipeline
    """
    del x
    annotations, patch_metadata = y
    return (mot_array_to_coco(annotations), patch_metadata)


def mot_coco_to_array(batch):
    """
    Map from extended coco format to 3D array (batch_size x detections x 9)

    An additional field, 'object_id', is required.
    If 'visibility' is not present, it defaults to 1 (visible)
    """
    if len(batch) == 0 or isinstance(batch[0], dict):
        not_batch = True
        batch = [batch]
    else:
        not_batch = False

    try:
        output = []
        for video in batch:
            rows = []
            for coco_dict in video:
                rows.append(
                    [
                        coco_dict["image_id"],
                        coco_dict["object_id"],
                        *coco_dict["bbox"],
                        coco_dict["score"],
                        coco_dict["category_id"],
                        coco_dict.get("visibility", 1),
                    ]
                )
            output.append(rows)
    except KeyError as e:
        raise KeyError(f"{e} is a required label key for multi-object tracking")

    if len(batch) == 1:
        output_array = np.array(output, dtype=np.float32)
    else:
        output_array = np.empty(len(batch), dtype=object)
        for i, out in enumerate(output):
            output_array[i] = np.array(out, dtype=np.float32)

    if not_batch:
        output_array = output_array[0]

    return output_array


def carla_multi_object_tracking_dev(
    split: str = "dev",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_video_tracking_canonical_preprocessing,
    label_preprocessing_fn=carla_mot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    max_frames: int = None,
    coco_format: bool = False,
    **kwargs,
):
    """
    Dev set for CARLA multi-object video tracking dataset, The dev set also contains green screens
    for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_multi_object_tracking_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_multi_object_tracking_dev batch size must be set to 1")

    if max_frames:
        clip = datasets.ClipFrames(max_frames)
        clip_labels = ClipMOTVideoTrackingLabels(max_frames)
    else:
        clip = None
        clip_labels = None

    if coco_format:
        coco_label_preprocess = mot_array_to_coco_label_preprocessing
    else:
        coco_label_preprocess = None

    preprocessing_fn = datasets.preprocessing_chain(clip, preprocessing_fn)

    label_preprocessing_fn = datasets.label_preprocessing_chain(
        mot_zero_index, clip_labels, coco_label_preprocess, label_preprocessing_fn
    )

    return datasets._generator_from_tfds(
        "carla_mot_dev:1.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=carla_video_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", ("annotations", "patch_metadata")),
        **kwargs,
    )


def carla_multi_object_tracking_test(
    split: str = "test",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = carla_video_tracking_canonical_preprocessing,
    label_preprocessing_fn=carla_mot_label_preprocessing,
    cache_dataset: bool = True,
    framework: str = "numpy",
    shuffle_files: bool = False,
    max_frames: int = None,
    coco_format: bool = False,
    **kwargs,
):
    """
    Test set for CARLA multi-object video tracking dataset, The test set also contains green screens
    for adversarial patch insertion.
    """
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the carla_multi_object_tracking_dev dataset"
        )
    if batch_size != 1:
        raise ValueError("carla_multi_object_tracking_dev batch size must be set to 1")

    if max_frames:
        clip = datasets.ClipFrames(max_frames)
        clip_labels = ClipMOTVideoTrackingLabels(max_frames)
    else:
        clip = None
        clip_labels = None

    if coco_format:
        coco_label_preprocess = mot_array_to_coco_label_preprocessing
    else:
        coco_label_preprocess = None

    preprocessing_fn = datasets.preprocessing_chain(clip, preprocessing_fn)

    label_preprocessing_fn = datasets.label_preprocessing_chain(
        mot_zero_index, clip_labels, coco_label_preprocess, label_preprocessing_fn
    )

    return datasets._generator_from_tfds(
        "carla_mot_test:1.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=carla_video_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", ("annotations", "patch_metadata")),
        **kwargs,
    )
