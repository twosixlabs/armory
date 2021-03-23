# from armory.utils.config_loading import load_model
from armory.baseline_models.pytorch.micronnet_gtsrb import get_art_model

import logging

logger = logging.getLogger(__name__)


class WrappedModel:
    def __init__(self, model_kwargs, wrapper_kwargs):
        self.model = get_art_model(model_kwargs, wrapper_kwargs)

    def defended_train(self, poisoned_data, gt, fit_kwargs):
        x_train, y_train = poisoned_data
        logger.info("Fitting final model...")
        self.model.fit(x_train, y_train, **fit_kwargs)
        # TODO: add confusion matrix report to return value
        return self.model, None


def get_defended_model(model_kwargs, wrapper_kwargs):
    wm = WrappedModel(model_kwargs, wrapper_kwargs)
    return wm
