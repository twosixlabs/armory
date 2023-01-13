"""
Resnet for speech commands classification.
"""

from typing import Optional

from art.estimators.classification import TensorFlowV2Classifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.layers import Lambda


def get_spectrogram(audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram  # shape (124, 129, 1)


def make_audio_resnet(**kwargs) -> tf.keras.Model:

    inputs = keras.Input(shape=(16000,))
    spectrogram = Lambda(lambda audio: get_spectrogram(audio))(inputs)

    resnet = keras.applications.ResNet50(
        weights=None,
        input_tensor=spectrogram,
        classes=12,
        **kwargs,
    )

    model = keras.Model(resnet.inputs, resnet.outputs)
    # ART's TensorFlowV2Classifier get_activations() requires a Sequential model
    model = keras.Sequential([model])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):

    if weights_path:
        raise ValueError(
            "This model is implemented for poisoning and does not (yet) load saved weights."
        )

    model = make_audio_resnet(**model_kwargs)

    loss_object = losses.SparseCategoricalCrossentropy()

    def train_step(model, samples, labels):
        with tf.GradientTape() as tape:
            predictions = model(samples, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    art_classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=12,
        input_shape=(16000,),
        **wrapper_kwargs,
    )

    return art_classifier
