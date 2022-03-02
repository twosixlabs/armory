"""
Inception_ResNet_v2 CNN model for 299x299x3 image classification
"""
from typing import Optional

from art.estimators.classification import KerasClassifier
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):
    input = tf.keras.Input(shape=(224, 224, 3))

    # Preprocessing layers
    img_scaled_to_255 = Lambda(lambda image: image * 255)(input)
    img_resized = Lambda(lambda image: tf.image.resize(image, (299, 299)))(
        img_scaled_to_255
    )
    # Model was trained with inputs normalized from -1 to 1
    img_normalized = Lambda(lambda image: (image / 127.5) - 1.0)(img_resized)

    inception_resnet_v2 = InceptionResNetV2(
        weights=None, input_tensor=img_normalized, **model_kwargs
    )
    model = Model(inputs=input, outputs=inception_resnet_v2.output)
    if weights_path:
        model.load_weights(weights_path)

    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
