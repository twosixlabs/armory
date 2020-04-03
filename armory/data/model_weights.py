"""
Utility to download model weights to cache.
"""


def download_all():
    from armory.baseline_models.keras.inception_resnet_v2 import get_art_model

    get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_file="inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5",
    )

    from armory.baseline_models.keras.resnet50 import get_art_model

    get_art_model(
        model_kwargs={},
        wrapper_kwargs={},
        weights_file="resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    )
