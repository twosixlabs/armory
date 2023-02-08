"""
CNN model for 28x28x1 image classification
"""
import tarfile

from art.estimators.classification import TFClassifier
import tensorflow.compat.v1 as tf

from armory import paths

tf.disable_eager_execution()
# TODO Update when ART is fixed with default_graph thing


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
    training_ph = tf.placeholder(tf.bool, shape=())

    x = tf.layers.conv2d(input_ph, filters=4, kernel_size=(5, 5), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, filters=10, kernel_size=(5, 5), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, 10)

    loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph)
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if weights_path:
        # Load Model using preferred save/restore method
        tar = tarfile.open(weights_path)
        tar.extractall(path=paths.runtime_paths().saved_model_dir)
        tar.close()
        # Restore variables...

    wrapped_model = TFClassifier(
        clip_values=(0.0, 1.0),
        input_ph=input_ph,
        output=logits,
        labels_ph=labels_ph,
        train=train_op,
        loss=loss,
        learning=training_ph,
        sess=sess,
        **wrapper_kwargs
    )

    return wrapped_model
