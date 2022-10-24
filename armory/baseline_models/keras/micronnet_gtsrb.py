"""
MicronNet CNN model for 48x48x3 image classification

Model contributed by: MITRE Corporation
"""
from typing import Optional

from art.estimators.classification import KerasClassifier
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Lambda,
    MaxPooling2D,
)

tf.compat.v1.disable_eager_execution()


def make_model(**kwargs) -> tf.keras.Model:
    # Model is based on MicronNet: https://arxiv.org/abs/1804.00497v3

    img_size = 48
    NUM_CLASSES = 43
    eps = 1e-6

    inputs = Input(shape=(img_size, img_size, 3))
    x = Lambda(lambda image: image * 255)(inputs)
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = Activation("relu")(x)
    x = Conv2D(29, (5, 5), padding="same")(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(59, (3, 3), padding="same")(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Conv2D(74, (3, 3), padding="same")(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Flatten()(x)
    x = Dense(300)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = Dense(300, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True
        ),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):
    model = make_model(**model_kwargs)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
