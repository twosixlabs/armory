"""
Inception_ResNet_v2 CNN model for 299x299x3 image classification
"""
from art.classifiers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from armory.data.utils import maybe_download_weights_from_s3


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
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
    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    wrapped_model = KerasClassifier(model, clip_values=(-1.0, 1.0), **wrapper_kwargs)
    return wrapped_model
