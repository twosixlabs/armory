"""
Super simple model for demonstrating resisc45 baseline
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier


def preprocessing_fn(img):
    return img.astype(np.float32) / 255.0


def make_model(**kwargs) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Conv2D(
            filters=4,
            kernel_size=(7, 7),
            strides=1,
            activation="relu",
            input_shape=(256, 256, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(
        Conv2D(
            filters=10,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(50, 50, 4),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(45, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.003),
        metrics=["accuracy"],
    )
    return model


def get_art_model(model_kwargs, wrapper_kwargs):
    model = make_model(**model_kwargs)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
