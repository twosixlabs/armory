"""
Standard preprocessing for different datasets
"""


import tensorflow as tf


REGISTERED_PREPROCESSORS = {}


def register(function, name=None):
    if name is None:
        name = function.__name__
    global REGISTERED_PREPROCESSORS
    REGISTERED_PREPROCESSORS[name] = function
    return function


def list_registered():
    return list(REGISTERED_PREPROCESSORS)


def get(name, version=None):
    if version is not None:
        raise NotImplementedError("version is not None for preprocessing to_canon")
    if name not in REGISTERED_PREPROCESSORS:
        raise KeyError(
            f"prepreprocessor {name} not registered. Use one of {list_registered()}"
        )
    return REGISTERED_PREPROCESSORS[name]


@register
def mnist(element):
    return (image_to_canon(element["image"]), element["label"])


@register
def digit(element):
    return (audio_to_canon(element["audio"]), element["label"])


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
