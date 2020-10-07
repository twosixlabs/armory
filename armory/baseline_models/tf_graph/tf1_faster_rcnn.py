"""
Implementing faster_rcnn_resnet50_coco from
TensorFlow1 Detection Model Zoo
(https://github.com/tensorflow/models/blob/master/research/
object_detection/g3doc/tf1_detection_zoo.md)
"""

from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
import tensorflow as tf

#from armory import paths
#from armory.data.utils import maybe_download_weights_from_s3


def preprocessing_fn(img):
    return img


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    images = tf.placeholder(tf.float32, shape=(1, 1024, 1024, 3)) # trade resolution for speed

    # There is no model wrapper for object detection
    # This is an MSCOCO pre-trained model that outputs 0-indexed classes,
    # whose (1-indexed) label map can be found at
    # https://github.com/tensorflow/models/blob/master/research
    # /object_detection/data/mscoco_label_map.pbtxt
    model = TensorFlowFasterRCNN(
        images,
        model = None,
        filename = 'faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        url = 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        sess = None,
        is_training = False,
        clip_values = (0, 1),
        channels_first = False,
        preprocessing = (0, 255),
        preprocessing_defences = None,  # model_kwargs?
        postprocessing_defences = None,  # model_kwargs?
        attack_losses = (  # model_kwargs?
            "Loss/RPNLoss/localization_loss",
            "Loss/RPNLoss/objectness_loss",
            "Loss/BoxClassifierLoss/localization_loss",
            "Loss/BoxClassifierLoss/classification_loss",
        )
    )
    return model
