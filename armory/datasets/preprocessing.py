"""
Standard preprocessing for different datasets
"""


import tensorflow as tf
from armory.data.adversarial.apricot_metadata import ADV_PATCH_MAGIC_NUMBER_LABEL_ID


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
def so2sat(element):
    # This preprocessing function assumes a so2sat builder_config of 'all' (i.e. multimodal)
    # as opposed to 'rgb'
    sentinel_1 = element["sentinel1"]
    sentinel_2 = element["sentinel2"]

    sar = sentinel_1[..., :4]
    sar /= 128.0

    eo = sentinel_2
    eo /= 4.0
    sar_eo_combined = tf.concat([sar, eo], axis=-1)
    return sar_eo_combined, element["label"]


@register
def digit(element):
    return (audio_to_canon(element["audio"]), element["label"])


@register
def carla_obj_det_test(element, modality="rgb"):
    return carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )


@register
def carla_obj_det_dev(element, modality="rgb"):
    return carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )


@register
def carla_obj_det_train(element, modality="rgb"):
    return carla_multimodal_obj_det(
        element["image"], modality=modality
    ), convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"])


@register
def carla_over_obj_det_dev(element, modality="rgb"):
    return carla_multimodal_obj_det(element["image"], modality=modality), (
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"]),
        element["patch_metadata"],
    )


def clip(batch, max_frames=None):
    if max_frames:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames {max_frames} must be > 0")
        batch = batch[:max_frames, :]
    return batch


def clip_labels(boxes, patch_metadata_dict, max_frames=None):
    if max_frames:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames {max_frames} must be > 0")
        boxes = boxes[:max_frames, :]
        patch_metadata_dict = {
            k: v[:max_frames, :] for (k, v) in patch_metadata_dict.items()
        }
    return boxes, patch_metadata_dict


def carla_video_tracking_preprocess(x, max_frames=None):
    x = clip(x, max_frames=max_frames)
    x = tf.cast(x, tf.float32) / 255.0
    return x


def carla_video_tracking_preprocess_labels(y, y_patch_metadata, max_frames=None):
    y, y_patch_metadata = clip_labels(y, y_patch_metadata, max_frames=max_frames)
    # Update labels
    y = {"boxes": y}
    y_patch_metadata = {
        k: (tf.squeeze(v, axis=0) if v.shape[0] == 1 else v)
        for k, v in y_patch_metadata.items()
    }
    return y, y_patch_metadata


def carla_video_tracking(element, max_frames=None):
    return carla_video_tracking_preprocess(
        element["video"],
        max_frames=max_frames,
    ), carla_video_tracking_preprocess_labels(
        element["bboxes"], element["patch_metadata"], max_frames=max_frames
    )


carla_video_tracking_dev = register(carla_video_tracking, "carla_video_tracking_dev")
carla_video_tracking_test = register(carla_video_tracking, "carla_video_tracking_test")


@register
def carla_over_obj_det_train(element, modality="rgb"):
    return carla_multimodal_obj_det(
        element["image"], modality=modality
    ), convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"])


def mot_zero_index(y, y_patch_metadata):
    if tf.rank(y) == 2:
        y = tf.tensor_scatter_nd_update(y, [[0, 0]], [y[0, 0] - 1])
    else:
        for i in tf.range(tf.shape(y)[0]):
            y_i = tf.expand_dims(y[i], axis=0)  # Add an extra dimension
            y_i = tf.tensor_scatter_nd_update(y_i, [[0, 0]], [y_i[0, 0] - 1])
            y_i = tf.squeeze(y_i, axis=0)  # Remove the extra dimension
            y = tf.tensor_scatter_nd_update(y, [[i]], [y_i])
    return y, y_patch_metadata


def mot_array_to_coco(batch):
    """
    Map from 3D array (batch_size x detections x 9) to extended coco format
        of dimension (batch_size x frames x detections_per_frame)
    NOTE: 'image_id' is given as the frame of a video, so is not unique
    """
    if len(batch.shape) == 2:
        not_batch = True
        batch = tf.expand_dims(batch, axis=0)
    elif len(batch.shape) == 3:
        not_batch = False
    else:
        raise ValueError(f"batch.ndim {len(batch.shape)} is not in (2, 3)")

    # output = tf.TensorArray(dtype=tf.float32, size=batch.shape[0], dynamic_size=False)
    output = []
    for i in range(batch.shape[0]):
        array = batch[i]
        # if not tf.math.greater(tf.shape(array)[0], 0):
        if array.shape[0] == 0:
            # no object detections
            # output = output.write(i, [])
            output.append([])
            continue

        # frames = tf.TensorArray(dtype=tf.float32, size=tf.shape(array)[0], dynamic_size=False)
        frames = []
        for detection in array:
            frame = tf.lookup.StaticHashTable(
                {
                    # TODO: should image_id include video number as well?
                    "image_id": tf.cast(tf.math.round(detection[0]), tf.int32),
                    "category_id": tf.cast(tf.math.round(detection[7]), tf.int32),
                    "bbox": tf.cast(detection[2:6], float),
                    "score": tf.cast(detection[6], float),
                    # The following are extended fields
                    "object_id": tf.cast(
                        tf.math.round(detection[1]), tf.int32
                    ),  # for a specific object across frames
                    "visibility": tf.cast(detection[8], float),
                }
            )
            frames.append(frame)
            # frames = frames.write(frames.size(), frame)
        # output = output.write(i, frames)
        output.append(frames)

    if not_batch:
        output = output[0]

    raise NotImplementedError("This does not work yet")
    return output


def carla_mot_label_preprocessing(
    y, y_patch_metadata, coco_format=False, max_frames=None
):
    y, y_patch_metadata = mot_zero_index(y, y_patch_metadata)
    y, y_patch_metadata = clip_labels(y, y_patch_metadata, max_frames=max_frames)
    if coco_format:
        y = mot_array_to_coco(y)
    y_patch_metadata = {k: tf.squeeze(v, axis=0) for k, v in y_patch_metadata.items()}
    return y, y_patch_metadata


@register
def carla_multi_object_tracking_dev(element, coco_format=False, max_frames=None):
    return carla_video_tracking_preprocess(
        element["video"],
        max_frames=max_frames,
    ), carla_mot_label_preprocessing(
        element["annotations"],
        element["patch_metadata"],
        coco_format=coco_format,
        max_frames=max_frames,
    )


@register
def xview(element):
    return image_to_canon(element["image"]), convert_tf_obj_det_label_to_pytorch(
        element["image"], element["objects"]
    )


@register
def apricot_dev(element):
    return image_to_canon(element["image"]), replace_magic_val(
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"])
    )


@register
def apricot_test(element):
    return image_to_canon(element["image"]), replace_magic_val(
        convert_tf_obj_det_label_to_pytorch(element["image"], element["objects"])
    )


def replace_magic_val(y):
    raw_adv_patch_category_id = 12
    rhs = y["labels"]
    y["labels"] = tf.where(
        tf.equal(rhs, raw_adv_patch_category_id),
        tf.ones_like(rhs, dtype=tf.int64) * ADV_PATCH_MAGIC_NUMBER_LABEL_ID,
        rhs,
    )
    return y


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
    This function converts COCO labels from a 0-79 range to the standard 1-90 with 10 unused indices
    (see https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt).
    The label map used matches the link above.
    """
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
