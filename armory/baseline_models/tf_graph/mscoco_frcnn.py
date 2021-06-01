"""
Implementing faster_rcnn_resnet50_coco from
TensorFlow1 Detection Model Zoo
(https://github.com/tensorflow/models/blob/master/research/
object_detection/g3doc/tf1_detection_zoo.md)
"""

from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
import tensorflow as tf


class TensorFlowFasterRCNNOneIndexed(TensorFlowFasterRCNN):
    """
    This is an MSCOCO pre-trained model. Note that the inherited TensorFlowFasterRCMM class
    outputs 0-indexed classes, while this wrapper class outputs 1-indexed classes. A label map can be found at
    https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt

    This model only performs inference and is not trainable. To train
    or fine-tune this model, please follow instructions at
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md
    """

    def __init__(self, images):
        super().__init__(
            images,
            model=None,
            filename="faster_rcnn_resnet50_coco_2018_01_28",
            url="http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz",
            sess=None,
            is_training=False,
            clip_values=(0, 1),
            channels_first=False,
            preprocessing_defences=None,
            postprocessing_defences=None,
            attack_losses=(
                "Loss/RPNLoss/localization_loss",
                "Loss/RPNLoss/objectness_loss",
                "Loss/BoxClassifierLoss/localization_loss",
                "Loss/BoxClassifierLoss/classification_loss",
            ),
        )

    def compute_loss(self, x, y):
        raise NotImplementedError


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    # APRICOT inputs should have shape (1, None, None, 3) while DAPRICOT inputs have shape
    # (3, None, None, 3)
    images = tf.placeholder(
        tf.float32, shape=(model_kwargs.get("batch_size", 1), None, None, 3)
    )
    model = TensorFlowFasterRCNNOneIndexed(images)
    return model
