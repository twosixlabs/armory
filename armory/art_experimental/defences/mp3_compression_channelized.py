from art.defences.preprocessor import Mp3Compression
import numpy as np


class Mp3CompressionChannelized(Mp3Compression):
    """
    Add a channel axis to the input
    """

    def __init__(
        self, sample_rate, channels_first=False, apply_fit=False, apply_predict=True,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )

    def __call__(self, x, y=None):
        # Check implicit batchsize
        if x.shape[0] == 1:
            raise NotImplementedError(
                "Batch size 1 currently not supported for Mp3CompressionChannelizedDefense"
            )

        # add a channel axis
        x = np.expand_dims(x, axis=-1)
        x, _ = super().__call__(x)
        x = np.squeeze(x)
        return x, y
