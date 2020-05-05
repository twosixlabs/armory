"""
ResNet50 CNN model for 244x244x3 image classification
"""

import numpy as np
from art.classifiers import KerasClassifier
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image

from armory.data.utils import maybe_download_weights_from_s3


IMAGENET_MEANS = [103.939, 116.779, 123.68]


def preprocess_input_resnet50(img):
    # Model was trained with inputs zero-centered on ImageNet mean
    img = img[..., ::-1]
    img[..., 0] -= IMAGENET_MEANS[0]
    img[..., 1] -= IMAGENET_MEANS[1]
    img[..., 2] -= IMAGENET_MEANS[2]

    return img


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (224, 224)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_resnet50(np.array(output))
    return output


def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = ResNet50(weights=None, **model_kwargs)

    if weights_file:
        filepath = maybe_download_weights_from_s3(weights_file)
        model.load_weights(filepath)

    wrapped_model = KerasClassifier(
        model,
        clip_values=(
            np.array(
                [
                    0.0 - IMAGENET_MEANS[0],
                    0.0 - IMAGENET_MEANS[1],
                    0.0 - IMAGENET_MEANS[2],
                ]
            ),
            np.array(
                [
                    255.0 - IMAGENET_MEANS[0],
                    255.0 - IMAGENET_MEANS[1],
                    255.0 - IMAGENET_MEANS[2],
                ]
            ),
        ),
        **wrapper_kwargs,
    )
    return wrapped_model
