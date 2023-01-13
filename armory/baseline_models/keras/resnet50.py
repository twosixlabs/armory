"""
ResNet50 CNN model for 244x244x3 image classification
"""
from typing import Optional

from art.estimators.classification import KerasClassifier
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()


IMAGENET_MEANS = [103.939, 116.779, 123.68]


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> KerasClassifier:
    """
    resnet pretrained on Imagenet. The initial layers transform the input from canonical form to the expected input
    format for the DenseNet-121.
    """
    input = tf.keras.Input(shape=(224, 224, 3))

    # Preprocessing layers
    img_scaled_to_255 = Lambda(lambda image: image * 255)(input)
    # Reorder image channels i.e. img = img[..., ::-1]
    img_channel_reorder = Lambda(lambda image: tf.reverse(image, axis=[-1]))(
        img_scaled_to_255
    )
    # Model was trained with inputs zero-centered on ImageNet mean
    img_normalized = Lambda(lambda image: image - IMAGENET_MEANS)(img_channel_reorder)

    resnet50 = ResNet50(weights=None, input_tensor=img_normalized, **model_kwargs)
    model = Model(inputs=input, outputs=resnet50.output)

    if weights_path:
        model.load_weights(weights_path)

    wrapped_model = KerasClassifier(
        model,
        clip_values=(0.0, 1.0),
        **wrapper_kwargs,
    )
    return wrapped_model
