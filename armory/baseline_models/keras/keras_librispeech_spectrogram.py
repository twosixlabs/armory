"""
Super simple model for demonstrating resisc45 baseline
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from art.classifiers import KerasClassifier

def preprocessing_fn(img):
    return img

def make_model(**kwargs) -> tf.keras.Model:
    a = Input(shape=(32,))
    b = Dense(32)(a)
    model = Model(inputs=a, outputs=b)
    return model

def get_art_model(model_kwargs, wrapper_kwargs):
    model = make_model(**model_kwargs)
    wrapped_model = KerasClassifier(model, clip_values=(0.0, 1.0), **wrapper_kwargs)
    return wrapped_model
