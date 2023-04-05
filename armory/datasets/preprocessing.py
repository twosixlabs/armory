"""
Standard preprocessing for different datasets
"""


import tensorflow as tf


REGISTERED_PREPROCESSORS = {}
DEFAULT = "DEFAULT"


def register(function, name=None):
    if name is None:
        name = function.__name__
    global REGISTERED_PREPROCESSORS
    REGISTERED_PREPROCESSORS[name] = function
    return function


def list_registered():
    return list(REGISTERED_PREPROCESSORS)


def get(name):
    if name not in REGISTERED_PREPROCESSORS:
        raise KeyError(
            f"prepreprocessor {name} not registered. Use one of {list_registered()}"
        )
    return REGISTERED_PREPROCESSORS[name]


def has(name):
    return name in REGISTERED_PREPROCESSORS


@register
def supervised_image_classification(element):
    return (image_to_canon(element["image"]), element["label"])


mnist = register(supervised_image_classification, "mnist")
cifar10 = register(supervised_image_classification, "cifar10")
cifar100 = register(supervised_image_classification, "cifar100")
imagenette = register(supervised_image_classification, "imagenette")
resisc45 = register(supervised_image_classification, "resisc45")


@register
def digit(element):
    return (audio_to_canon(element["audio"]), element["label"])


@register
def carla_obj_det_dev(element, modality="rgb"):
    return carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )


@register
def carla_over_obj_det_dev(element, modality="rgb"):
    return carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )


@register
def xview(element):
    return image_to_canon(element["image"]), convert_tf_obj_det_label_to_pytorch(
        element["image"], element["objects"]
    )


def image_to_canon(image, resize=None, target_dtype=tf.float32, input_type="uint8"):
    """
    TFDS Image feature uses (height, width, channels)
    """
    if input_type == "uint8":
        scale = 255.0
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")
    image = tf.cast(image, target_dtype)
    image = image / scale
    if resize is not None:
        resize = tuple(resize)
        if len(resize) != 2:
            raise ValueError(f"resize must be None or a 2-tuple, not {resize}")
        image = tf.image.resize(image, resize)
    return image


def audio_to_canon(audio, resample=None, target_dtype=tf.float32, input_type="int16"):
    """
    Note: input_type is the scale of the actual data
        TFDS typically converts to tf.inf64, which is not helpful in this case
    """
    if input_type == "int16":
        scale = 2**15
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")
    audio = tf.cast(audio, target_dtype)
    audio = audio / scale
    if resample is not None:
        raise NotImplementedError("resampling not currently supported")
    return audio


def video_to_canon(
    video,
    resize=None,
    target_dtype=tf.float32,
    input_type="uint8",
    max_frames: int = None,
):
    """
    TFDS Video feature uses (num_frames, height, width, channels)
    """
    if input_type == "uint8":
        scale = 255.0
    else:
        raise NotImplementedError(f"Currently only supports uint8, not {input_type}")

    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        video = video[:max_frames]
    video = tf.cast(video, target_dtype)
    video = video / scale
    if resize is not None:
        raise NotImplementedError("resizing video")
    return video


def carla_multimodal_obj_det(x, modality="rgb"):
    if modality == "rgb":
        return image_to_canon(x[0])
    elif modality == "depth":
        return image_to_canon(x[1])
    elif modality == "both":
        return image_to_canon(tf.concat([x[0], x[1]], axis=-1))
    else:
        raise ValueError(
            f"Found unexpected modality {modality}. Expected one of ('rgb', 'depth', 'both')."
        )


def convert_tf_boxes_to_pytorch(x, box_array):
    """
    Converts object detection boxes from TF format of [y1/height, x1/width, y2/height, x2/width]
    to PyTorch format of [x1, y1, x2, y2]

    :param x: TF tensor of shape (nb, H, W, C) or (H, W, C)
    :param box_array: TF tensor of shape (num_boxes, 4)
    :return: TF tensor of shape (num_boxes, 4)
    """
    x_shape = tf.shape(x)
    if len(x_shape) == 3:
        height = x_shape[0]
        width = x_shape[1]
    elif len(x_shape) == 4:
        height = x_shape[1]
        width = x_shape[2]
    else:
        raise ValueError(f"Received unexpected shape {x.shape}")
    # reorder [y1/height, x1/width, y2/height, x2/width] to [x1/width, y1/height, x2/width, y2/height]
    converted_boxes = tf.gather(box_array, [1, 0, 3, 2], axis=1)

    # un-normalize boxes
    converted_boxes *= [width, height, width, height]
    return converted_boxes


def convert_tf_obj_det_label_to_pytorch(x, y_object):
    # raw dataset has boxes in TF format of [y1/height, x1/width, y2/height, x2/width]
    box_array_tf_format = y_object["boxes"]
    y_object["boxes"] = convert_tf_boxes_to_pytorch(x, box_array_tf_format)
    return y_object


def infer_from_dataset_info(info, split):
    raise NotImplementedError


@register
def coco(element):
    return image_to_canon(element["image"]), coco_label_preprocessing(
        element["image"], element["objects"]
    )


def coco_label_preprocessing(x, y):
    """
    If batch_size is 1, this function converts the single y dictionary to a list of length 1.
    This function converts COCO labels from a 0-79 range to the standard 0-89 with 10 unused indices
    (see https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt).
    The label map used matches the link above, with the note that labels start from 0 rather than 1.
    """
    # # This will be true only when batch_size is 1
    # if isinstance(y, dict):
    #     y = [y]
    idx_map = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 10,
        10: 11,
        11: 13,
        12: 14,
        13: 15,
        14: 16,
        15: 17,
        16: 18,
        17: 19,
        18: 20,
        19: 21,
        20: 22,
        21: 23,
        22: 24,
        23: 25,
        24: 27,
        25: 28,
        26: 31,
        27: 32,
        28: 33,
        29: 34,
        30: 35,
        31: 36,
        32: 37,
        33: 38,
        34: 39,
        35: 40,
        36: 41,
        37: 42,
        38: 43,
        39: 44,
        40: 46,
        41: 47,
        42: 48,
        43: 49,
        44: 50,
        45: 51,
        46: 52,
        47: 53,
        48: 54,
        49: 55,
        50: 56,
        51: 57,
        52: 58,
        53: 59,
        54: 60,
        55: 61,
        56: 62,
        57: 63,
        58: 64,
        59: 65,
        60: 67,
        61: 70,
        62: 72,
        63: 73,
        64: 74,
        65: 75,
        66: 76,
        67: 77,
        68: 78,
        69: 79,
        70: 80,
        71: 81,
        72: 82,
        73: 84,
        74: 85,
        75: 86,
        76: 87,
        77: 88,
        78: 89,
        79: 90,
    }

    keys, values = list(zip(*idx_map.items()))
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys, dtype=tf.int64),
            values=tf.constant(values),
        ),
        default_value=tf.constant(-1),
        name="object_index",
    )

    y["boxes"] = y.pop("bbox")
    y = convert_tf_obj_det_label_to_pytorch(x, y)
    y["labels"] = table.lookup(y.pop("label"))
    return y
