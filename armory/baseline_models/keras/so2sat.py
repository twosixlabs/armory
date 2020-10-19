from tensorflow import slice
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Input, concatenate, Lambda
from art.estimators.classification import KerasClassifier
from tensorflow.keras.optimizers import SGD


def make_model(**kwargs):
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
    renorm_input = Lambda(lambda x: x * 115.25348)(concat_input)

    SAR_input = Lambda(
        lambda concat_input: slice(concat_input, [0, 0, 0, 0], [-1, 32, 32, 4])
    )(renorm_input)
    EO_input = Lambda(
        lambda concat_input: slice(concat_input, [0, 0, 0, 4], [-1, 32, 32, 10])
    )(renorm_input)

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


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    model = make_model(**model_kwargs)
    if weights_path:
        model.load_weights(weights_path)

    wrapped_model = KerasClassifier(model=model, **wrapper_kwargs)
    return wrapped_model
