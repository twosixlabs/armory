"""
Utility to download model weights to cache.
"""


def download_all():
    from armory.baseline_models.keras.inception_resnet_v2 import get_art_model

    get_art_model(model_kwargs={"weights": "imagenet"}, wrapper_kwargs={})

    from armory.baseline_models.keras.resnet50 import get_art_model

    get_art_model(model_kwargs={"weights": "imagenet"}, wrapper_kwargs={})
