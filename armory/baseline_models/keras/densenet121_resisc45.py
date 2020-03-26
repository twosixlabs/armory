import functools
import os

from art.classifiers import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from armory import paths

num_classes = 45


@functools.lru_cache(maxsize=1)
def mean_std():
    resisc_mean = np.load(
        os.path.join(
            paths.docker().dataset_dir, "resisc45_split/3.0.0/resisc-45_rgb_means.npy"
        )
    )
    resisc_std = np.load(
        os.path.join(
            paths.docker().dataset_dir, "resisc45_split/3.0.0/resisc-45_rgb_stdevs.npy"
        )
    )
    return resisc_mean, resisc_std


def preprocess_input_densenet121_resisc(img):
    # Model was trained with Caffe preprocessing on the images
    # load the mean and std of the [0,1] normalized dataset
    # Normalize images: divide by 255 for [0,1] range
    mean, std = mean_std()
    img_norm = img / 255.0
    # Divide by normalized channel means from rgb_means.npy
    # divide each channel by normalized channel stdevs from rgb_stdevs.npy
    output_img = (img_norm - mean) / std
    return output_img


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (224, 224)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_densenet121_resisc(np.array(output))
    return output


def make_densenet121_resisc_model(**model_kwargs) -> tf.keras.Model:
    # Load ImageNet pre-trained DenseNet
    model_notop = DenseNet121(
        include_top=False, weights=None, input_shape=(224, 224, 3)
    )

    # Add new layers
    x = GlobalAveragePooling2D()(model_notop.output)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create graph of new model and freeze pre-trained layers
    new_model = Model(inputs=model_notop.input, outputs=predictions)

    for layer in new_model.layers[:-1]:
        layer.trainable = False
        if "bn" == layer.name[-2:]:  # allow batchnorm layers to be trainable
            layer.trainable = True

    # compile the model
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    pretrained = model_kwargs["pretrained"]
    if pretrained:
        # load weights into a model loaded from pre-trained
        filepath = os.path.join(
            paths.docker().saved_model_dir, "DenseNet121", model_kwargs["model_file"]
        )
        new_model.load_weights(filepath)
    return new_model


def get_art_model(model_kwargs, wrapper_kwargs):
    model = make_densenet121_resisc_model(**model_kwargs)
    mean, std = mean_std()
    wrapped_model = KerasClassifier(
        model, clip_values=((0.0 - mean) / std, (1.0 - mean) / std), **wrapper_kwargs
    )
    return wrapped_model
