from art.classifiers import KerasClassifier
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as preprocess_input_inception_resnet_v2,
)


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (299, 299)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_inception_resnet_v2(np.array(output))
    return output


def get_art_model(model_kwargs, wrapper_kwargs):
    model = InceptionResNetV2(**model_kwargs)
    wrapped_model = KerasClassifier(model, **wrapper_kwargs)
    return wrapped_model
