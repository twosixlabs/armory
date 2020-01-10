"""

"""
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier

keras.backend.clear_session()


def make_mnist_model() -> keras.Model:
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
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=0.003),
        metrics=["accuracy"],
    )
    return model


MODEL = KerasClassifier(make_mnist_model())
