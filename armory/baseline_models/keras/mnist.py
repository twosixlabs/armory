"""

"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier

from armory import paths
from armory.data.utils import download_file_from_s3


def preprocessing_fn(img):
    # Model will trained with inputs normalized from 0 to 1
    img = img.astype(np.float32) / 255.0
    return img


def make_mnist_model(**kwargs) -> tf.keras.Model:
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


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = make_mnist_model(**model_kwargs)
    if weights_file:
        saved_model_dir = paths.docker().saved_model_dir
        filepath = os.path.join(saved_model_dir, weights_file)

        if not os.path.isfile(filepath):
            download_file_from_s3(
                "armory-public-data",
                f"model-weights/{weights_file}",
                f"{saved_model_dir}/{weights_file}",
            )

        model.load_weights(filepath)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
