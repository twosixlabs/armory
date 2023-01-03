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


@register
def librispeech(element, audio_kwargs=None):
    # TODO: determine how to fix np.array([<byte>], dtype=object) output for text
    #    https://github.com/tensorflow/tensorflow/issues/34871
    #    Our traditional behavior to decode to str once in numpy
    #    This can be done via: y.astype("U")
    #    Currently, this is handled by scenarios or metrics after dataset output
    # NOTE: 16000 sampling rate
    if audio_kwargs is None:
        audio_kwargs = {}
    text = element["text"]
    speech = audio_to_canon(element["speech"], **audio_kwargs)
    return (speech, text)


librispeech_dev_test = register(librispeech, "librispeech_dev_test")


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
