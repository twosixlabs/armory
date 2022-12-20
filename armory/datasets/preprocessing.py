"""
Standard preprocessing for different datasets
"""


import tensorflow as tf
from armory.datasets.context import contexts


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
    x = carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )
    print(x)
    return x


@register
def carla_video_tracking_dev(element, max_frames=None):
    return carla_video_tracking(
        element["video"], max_frames=max_frames, split="dev"
    ), carla_video_tracking_labels(
        element["video"],
        (element["bboxes"], element["patch_metadata"]),
        max_frames=max_frames,
    )


@register
def carla_video_tracking_test(element, max_frames=None):
    return carla_video_tracking(
        element["video"], max_frames=max_frames, split="test"
    ), carla_video_tracking_labels(
        element["video"],
        (element["bboxes"], element["patch_metadata"]),
        max_frames=max_frames,
    )


@register
def xview(element):
    return image_to_canon(element["image"]), convert_tf_obj_det_label_to_pytorch(
        element["image"], element["objects"]
    )


mnist = register(supervised_image_classification, "mnist")
cifar10 = register(supervised_image_classification, "cifar10")
resisc45 = register(supervised_image_classification, "resisc45")


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


def preprocessing_chain(*args):
    """
    Wraps and returns a sequence of preprocessing functions
    """
    functions = [x for x in args if x is not None]
    if not functions:
        return None

    def wrapped(x):
        for function in functions:
            x = function(x)
        return x

    return wrapped


def label_preprocessing_chain(*args):
    """
    Wraps and returns a sequence of label preprocessing functions.
    Note that this function differs from preprocessing_chain() in that
    it chains across (x, y) instead of just x
    """
    functions = [x for x in args if x is not None]
    if not functions:
        return None

    def wrapped(x, y):
        for function in functions:
            y = function(x, y)
        return y

    return wrapped


def check_shapes(actual, target):
    """
    Ensure that shapes match, ignoring None values

    actual and target should be tuples
    """
    if type(actual) != tuple:
        raise ValueError(f"actual shape {actual} is not a tuple")
    if type(target) != tuple:
        raise ValueError(f"target shape {target} is not a tuple")
    if len(actual) != len(target):
        raise ValueError(f"len(actual) {len(actual)} != len(target) {len(target)}")
    for a, t in zip(actual, target):
        if a != t and t is not None:
            raise ValueError(f"shape {actual} does not match shape {target}")


def canonical_variable_image_preprocess(context, batch):
    """
    Preprocessing when images are of variable size
    """
    if batch.dtype != context.input_type:
        if batch.dtype == object:
            raise NotImplementedError(
                "<object> dtype not yet supported for variable image processing."
            )
        raise ValueError(
            f"input dtype {batch.dtype} not in ({context.input_type}, 'O')"
        )
    check_shapes(tuple(batch.shape), context.x_shape)
    assert batch.dtype == context.input_type
    return batch


class ClipFrames:
    """
    Clip Video Frames
        Assumes first two dims are (batch, frames, ...)
    """

    def __init__(self, max_frames):
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames {max_frames} must be > 0")
        self.max_frames = max_frames

    def __call__(self, batch):
        return batch[:, : self.max_frames]


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


def carla_video_tracking(x, split, max_frames=None):
    clip = ClipFrames(max_frames) if max_frames else None
    preprocessing_fn = preprocessing_chain(
        clip, lambda batch: canonical_variable_image_preprocess(
            contexts[f"carla_video_tracking_{split}"], batch
        )
    )
    return preprocessing_fn(x)


def carla_video_tracking_label_preprocessing(x, y):
    box_labels, patch_metadata = y
    box_array = (
        tf.squeeze(box_labels, axis=0) if box_labels.shape[0] == 1 else box_labels
    )
    box_labels = {"boxes": box_array}
    patch_metadata = {
        k: (tf.squeeze(v, axis=0) if v.shape[0] == 1 else v)
        for k, v in patch_metadata.items()
    }
    return box_labels, patch_metadata


def carla_video_tracking_labels(x, y, max_frames):
    clip_labels = ClipVideoTrackingLabels(max_frames) if max_frames else None
    label_preprocessing_fn = label_preprocessing_chain(
        clip_labels, carla_video_tracking_label_preprocessing
    )
    return label_preprocessing_fn(x, y)


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
