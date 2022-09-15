"""
CNN for speech commands classification.  
Model and spectrogram function from https://www.tensorflow.org/tutorials/audio/simple_audio
"""

from typing import Optional


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from art.estimators.classification import TensorFlowV2Classifier


def get_spectrogram(audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram  # shape (124, 129, 1)


def make_audio_cnn(**kwargs) -> tf.keras.Model:
    """
    A CNN for audio poisoning on the speech commands dataset
    """
    norm_layer = layers.Normalization()
    input_shape = (16000,)
    num_labels = 12
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Lambda(get_spectrogram),
            layers.Resizing(32, 32),
            norm_layer,
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ]
    )

    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):

    if weights_path:
        raise ValueError(
            "This model is implemented for poisoning and does not (yet) load saved weights."
        )

    model = make_audio_cnn(**model_kwargs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    art_classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        input_shape=(16000,),
        nb_classes=12,
        **wrapper_kwargs,
    )

    return art_classifier
