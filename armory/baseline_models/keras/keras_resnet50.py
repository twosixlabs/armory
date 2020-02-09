from art.classifiers import KerasClassifier
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

preprocessing_fn = preprocess_input


def get_art_model(model_kwargs, wrapper_kwargs):
    model = ResNet50(**model_kwargs)
    wrapped_model = KerasClassifier(model, **wrapper_kwargs)
    return wrapped_model
