"""
CNN model for 28x28x1 image classification
"""
from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()


def make_mnist_model(**kwargs) -> tf.keras.Model:
    """
    This is a simple CNN for MNIST and does not achieve SotA performance
    """
    model = Sequential()
    model.add(
        Conv2D(
            filters=4,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        Conv2D(
            filters=10,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(23, 23, 4),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.003),
        metrics=["accuracy"],
    )
    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):
    model = make_mnist_model(**model_kwargs)
    if weights_path:
        model.load_weights(weights_path)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
