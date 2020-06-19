"""
This module implements adversarial training following Madry's Protocol.
| Paper link: https://arxiv.org/abs/1706.06083
| Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
    principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
    evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).

This has been modified from ART by adding a fit_generator functionality
"""

import logging

from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescent

logger = logging.getLogger(__name__)


class AdversarialTrainerMadryPGD(Trainer):
    """
    Class performing adversarial training following Madry's Protocol.
    Paper link: https://arxiv.org/abs/1706.06083
    Please keep in mind the limitations of defences. While adversarial training is
    widely regarded as a promising, principled approach to making classifiers more
    robust (see https://arxiv.org/abs/1802.00420), very careful evaluations are
    required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(self, classifier, eps=0.03, eps_step=0.008, max_iter=7, ratio=1.0):
        self.attack = ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps_step, max_iter=max_iter,
        )

        self.trainer = AdversarialTrainer(classifier, self.attack, ratio=ratio)

    def fit(self, x, y, **kwargs):
        self.trainer.fit(x, y, **kwargs)

    def fit_generator(self, generator, nb_epochs, **kwargs):
        self.trainer.fit_generator(generator, nb_epochs=nb_epochs, **kwargs)

    def get_classifier(self):
        return self.trainer.get_classifier()
