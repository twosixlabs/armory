"""
MicronNet CNN model for 48x48x3 image classification

Model contributed by: MITRE Corporation
"""
import numpy as np
from PIL import ImageOps, Image
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, Activation
from tensorflow.keras.layers import Flatten, BatchNormalization, MaxPooling2D
from art.classifiers import KerasClassifier


def preprocessing_fn(img):
    img_size = 48
    img_out = []
    for im in img:
        img_eq = ImageOps.equalize(Image.fromarray(im))
        width, height = img_eq.size
        min_side = min(img_eq.size)
        center = width // 2, height // 2

        left = center[0] - min_side // 2
        top = center[1] - min_side // 2
        right = center[0] + min_side // 2
        bottom = center[1] + min_side // 2

        img_eq = img_eq.crop((left, top, right, bottom))
        img_eq = np.array(img_eq.resize([img_size, img_size]))

        img_out.append(img_eq)

    return np.array(img_out, dtype=np.float32)


def make_model(**kwargs) -> tf.keras.Model:
    # Model is based on MicronNet: https://arxiv.org/abs/1804.00497v3

    img_size = 48
    NUM_CLASSES = 43
    eps = 1e-6

    inputs = Input(shape=(img_size, img_size, 3))
    x = Conv2D(1, (1, 1), padding="same")(inputs)
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


def get_art_model(model_kwargs, wrapper_kwargs, weights=None):
    model = make_model(**model_kwargs)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 255.0), **wrapper_kwargs)
    return wrapped_model
