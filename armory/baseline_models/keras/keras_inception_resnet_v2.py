from art.classifiers import KerasClassifier
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)

preprocessing_fn = preprocess_input
MODEL = KerasClassifier(InceptionResNetV2(weights="imagenet"))
