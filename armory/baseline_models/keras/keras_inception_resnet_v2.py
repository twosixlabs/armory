from art.classifiers import KerasClassifier
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)

preprocessing_fn = preprocess_input


def get_art_model(model_kwargs, wrapper_kwargs):
    model = InceptionResNetV2(**model_kwargs)
    wrapped_model = KerasClassifier(model, **wrapper_kwargs)
    return wrapped_model
