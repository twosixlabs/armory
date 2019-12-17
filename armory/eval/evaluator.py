"""

"""

import numpy as np
from armory.webapi.data import SUPPORTED_DATASETS
from art.classifiers.classifier import Classifier
from art.attacks import FastGradientMethod

import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install()


# TODO: Preprocessing needs to be refactored elsewhere!
def _normalize_img(img_batch):
    norm_batch = img_batch.astype(np.float32) / 255.
    return norm_batch


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self._verify_config()

    def _verify_config(self):
        assert isinstance(self.config, dict)
        assert isinstance(self.config["model"], Classifier)

    def test_classifer(self):
        if self.config["data"] not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Configured data {self.config['data']} not found in"
                f" supported datasets: {list(SUPPORTED_DATASETS.keys())}"
            )

        classifier = self.config["model"]
        train_ds, test_ds = SUPPORTED_DATASETS[self.config["data"]]()

        x_train, y_train = train_ds["image"], train_ds["label"]
        x_test, y_test = test_ds["image"], test_ds["label"]
        x_train, x_test = _normalize_img(x_train), _normalize_img(x_test)

        classifier.fit(
            x_train,
            y_train,
            batch_size=64,
            nb_epochs=3,
        )

        # Evaluate the ART classifier on benign test examples
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

        # Generate adversarial test examples
        attack = FastGradientMethod(classifier=classifier, eps=0.2)
        x_test_adv = attack.generate(x=x_test)

        # Evaluate the ART classifier on adversarial test examples
        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        print('Accuracy on adversarial test examples: {}%'.format(accuracy * 100))
