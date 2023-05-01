from copy import deepcopy
from typing import Optional

from art.estimators.classification import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow import slice
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.optimizers import SGD

tf.compat.v1.disable_eager_execution()


def make_model(**kwargs):
    """
    This is a simple CNN for So2SAT. The data are split into SAR and EO data streams and fed into their respective
    feature extraction networks. In the final layer, the two networks are fused to produce a single prediction output.
    """

    SAR_model = Sequential()
    SAR_model.add(
        Conv2D(18, kernel_size=(9, 9), activation="relu", input_shape=[32, 32, 4])
    )
    SAR_model.add(MaxPooling2D(pool_size=(2, 2)))
    SAR_model.add(Conv2D(36, (5, 5), activation="relu"))
    SAR_model.add(MaxPooling2D(pool_size=(2, 2)))

    EO_model = Sequential()
    EO_model.add(
        Conv2D(18, kernel_size=(9, 9), activation="relu", input_shape=[32, 32, 10])
    )
    EO_model.add(MaxPooling2D(pool_size=(2, 2)))
    EO_model.add(Conv2D(36, (5, 5), activation="relu"))
    EO_model.add(MaxPooling2D(pool_size=(2, 2)))

    input_shape = [32, 32, 14]
    concat_input = Input(input_shape)

    SAR_slice = Lambda(
        lambda concat_input: slice(concat_input, [0, 0, 0, 0], [-1, 32, 32, 4])
    )(concat_input)
    EO_slice = Lambda(
        lambda concat_input: slice(concat_input, [0, 0, 0, 4], [-1, 32, 32, 10])
    )(concat_input)

    SAR_input = Lambda(lambda x: x * 128)(SAR_slice)
    EO_input = Lambda(lambda x: x * 4)(EO_slice)

    encoded_SAR = SAR_model(SAR_input)
    encoded_EO = EO_model(EO_input)
    Concat = concatenate(axis=3, inputs=[encoded_SAR, encoded_EO])
    ConvFusion = Conv2D(120, (4, 4), activation="relu")(Concat)
    flat = Flatten()(ConvFusion)
    prediction = Dense(17, activation="softmax")(flat)
    fused_net = Model(inputs=concat_input, outputs=prediction)

    opt = SGD(learning_rate=0.01)
    fused_net.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    return fused_net


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):
    model = make_model(**model_kwargs)
    if weights_path:
        model.load_weights(weights_path)

    wrapper_kwargs2 = deepcopy(wrapper_kwargs)
    if "clip_values" in wrapper_kwargs:
        if isinstance(wrapper_kwargs["clip_values"], list):
            wrapper_kwargs2["clip_values"] = np.array(wrapper_kwargs2["clip_values"])

    wrapped_model = KerasClassifier(model=model, **wrapper_kwargs2)
    return wrapped_model
