from art.classifiers import KerasClassifier
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

preprocessing_fn = preprocess_input
MODEL = KerasClassifier(ResNet50(weights="imagenet"))
