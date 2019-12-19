import tensorflow as tf
from art.classifiers import KerasClassifier

RESNET50 = KerasClassifier(tf.keras.applications.resnet50.ResNet50())
