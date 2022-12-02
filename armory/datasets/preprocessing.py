"""
Standard preprocessing for different datasets

These modify, in tensorflow, element dicts and should output updated dicts
    Ideally, element keys should not be modified here
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


def audio(element, **kwargs):
    return {
        k: audio_to_canon(v, **kwargs) if k == "audio" else v
        for k, v in element.items()
    }


def image(element, **kwargs):
    return {
        k: image_to_canon(v, **kwargs) if k == "image" else v
        for k, v in element.items()
    }


def video(element, **kwargs):
    return {
        k: video_to_canon(v, **kwargs) if k == "video" else v
        for k, v in element.items()
    }


digit = register(audio, "digit")
mnist = register(image, "mnist")
cifar10 = register(image, "cifar10")


@register
def carla_over_obj_det_dev(element, modality="rgb"):
    out = {}
    out["image"] = carla_over_obj_det_image(element["image"], modality=modality)
    out["objects"], out["patch_metadata"] = carla_over_obj_det_dev_label(
        element["image"], element["objects"], element["patch_metadata"]
    )
    return out


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


def carla_over_obj_det_image(x, modality="rgb"):
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


def carla_over_obj_det_dev_label(x, y_object, y_patch_metadata):
    # convert from TF format of [y1/height, x1/width, y2/height, x2/width] to PyTorch format
    # of [x1, y1, x2, y2]
    height, width = x.shape[1:3]  # TODO: better way to extract width/height?

    # reorder [y1/height, x1/width, y2/height, x2/width] to [x1/width, y1/height, x2/width, y2/height]
    converted_boxes = tf.gather(y_object["boxes"], [1, 0, 3, 2], axis=1)

    # un-normalize boxes
    converted_boxes *= [width, height, width, height]

    y_object["boxes"] = converted_boxes
    return y_object, y_patch_metadata


def infer_from_dataset_info(info, split):
    raise NotImplementedError
