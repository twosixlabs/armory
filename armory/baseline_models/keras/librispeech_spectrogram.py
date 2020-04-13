"""
CNN model for 241x100x1 audio spectrogram classification

Model contributed by: MITRE Corporation
"""
import numpy as np
from scipy import signal

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from art.classifiers import KerasClassifier

from armory.data.utils import maybe_download_weights_from_s3


def preprocessing_fn(audios):
    # -- SPECTROGRAM PARAMETERS --#
    samplerate_hz = 16000  # sample rate for wav files
    window_time_secs = 0.030  # 30 ms windows for frequency transforms
    num_samples_overlap = int(0.025 * samplerate_hz)  # 25 ms of overlap

    # -- NORMALIZATION PARAMETERS --#
    zscore_mean = -0.7731548539849517
    zscore_std = 3.5610712683198624
    scale_max = 15.441861
    scale_min = -4.6051702

    # Construct window
    window_num_samples = int(window_time_secs * samplerate_hz)
    window = signal.get_window(("tukey", 0.25), window_num_samples)

    def normalize_spectrogram(s):
        """ Normalize spectrogram s:
        1. s_ = np.log(s + 0.01)
        2. s_ = zscores(s_)
        3. s_ = minmax_scale(s_)

        Return normalized spectrogram s_ in range [-1, 1] with mean ~ 0 and std ~ 1
        """
        s_ = np.log(s + 0.01)
        s_ = (s_ - zscore_mean) / zscore_std
        s_ = (s_ - scale_min) / (scale_max - scale_min)
        return s_

    def spectrogram_241(samples):
        """ Return vector of frequences (f), vector of times (t), and 2d matrix spectrogram (s)
        for input audio samples.
        """
        # Construct spectrogram (f = frequencies array, t = times array, s = 2d spectrogram [f x t])
        f, t, s = signal.spectrogram(
            samples, samplerate_hz, window=window, noverlap=num_samples_overlap
        )

        # Normalize spectrogram
        s = normalize_spectrogram(s)

        return f, t, s

    if audios.dtype == np.int64:
        audios = [audios]

    outputs = []
    for aud in audios:
        aud = np.squeeze(aud)
        _, _, s = spectrogram_241(aud)
        outputs.append(s)

    return outputs


def make_model(**kwargs) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=1,
            activation="relu",
            input_shape=(241, 100, 1),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(40, activation="softmax"))

    model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=0.0002),
        metrics=["accuracy"],
    )

    return model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    model = make_model(**model_kwargs)
    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    wrapped_model = KerasClassifier(model, clip_values=(-1.0, 1.0), **wrapper_kwargs)
    return wrapped_model
