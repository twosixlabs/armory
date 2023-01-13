"""
DenseNet121 CNN model for 244x244x3 image classification

Model contributed by: MITRE Corporation
"""
from typing import Optional

from art.estimators.classification import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

tf.compat.v1.disable_eager_execution()

num_classes = 45


def mean_std():
    resisc_mean = np.array(
        [
            0.36386173189316956,
            0.38118692953271804,
            0.33867067558870334,
        ]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std


def make_densenet121_resisc_model(**model_kwargs) -> tf.keras.Model:
    """
    This model is an ImageNet-pretrained DenseNet-121 that is fine-tuned on RESISC-45.
    The initial layers transform the input from canonical form to the expected input
    format for the DenseNet-121.
    """
    input = tf.keras.Input(shape=(256, 256, 3))

    # Preprocessing layers
    img_scaled_to_255 = Lambda(lambda image: image * 255)(input)
    img_resized = Lambda(lambda image: tf.image.resize(image, (224, 224)))(
        img_scaled_to_255
    )
    img_scaled_to_1 = Lambda(lambda image: image / 255)(img_resized)
    mean, std = mean_std()
    img_standardized = Lambda(lambda image: (image - mean) / std)(img_scaled_to_1)

    # Load ImageNet pre-trained DenseNet
    model_notop = DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=img_standardized,
        input_shape=(224, 224, 3),
    )

    # Add new layers
    x = GlobalAveragePooling2D()(model_notop.output)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create graph of new model and freeze pre-trained layers
    new_model = Model(inputs=input, outputs=predictions)

    for layer in new_model.layers[:-1]:
        layer.trainable = False
        if "bn" == layer.name[-2:]:  # allow batchnorm layers to be trainable
            layer.trainable = True

    # compile the model
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return new_model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> KerasClassifier:
    model = make_densenet121_resisc_model(**model_kwargs)
    if weights_path:
        model.load_weights(weights_path)

    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
