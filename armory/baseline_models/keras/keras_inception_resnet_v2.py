from art.classifiers import KerasClassifier
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


def preprocess_input_inception_resnet_v2(img):
    # Model was trained with inputs normalized from -1 to 1
    img /= 127.5
    img -= 1.0
    return img


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
    wrapped_model = KerasClassifier(model, clip_values=(-1.0, 1.0), **wrapper_kwargs)
    return wrapped_model
