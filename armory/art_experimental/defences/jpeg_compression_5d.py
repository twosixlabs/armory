from art.defences import JpegCompression
import numpy as np


class JpegCompression5D(JpegCompression):
    def __init__(
        self,
        clip_values,
        quality=50,
        channel_index=3,
        apply_fit=True,
        apply_predict=False,
        transpose=None,
        means=None,
    ):
        super().__init__(
            clip_values,
            quality=quality,
            channel_index=channel_index,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )

        if transpose is None:
            transpose = (0, 1, 2, 3)  # identify operation
        if len(transpose) != 4:
            raise ValueError("transpose must have 4 dimensions specified")
        self.transpose = transpose

        if means is None:
            means = (0.0, 0.0, 0.0)  # identity operation
        if len(means) != 3:
            raise ValueError("means must have 3 values, one per channel")
        self.means = means

    def __call__(self, x, y=None):
        x_new = []
        for x_i in x:
            x_i = x_i.transpose(self.transpose)
            x_i = x_i + self.means  # use broadcasting
            if self.clip_values:
                np.clip(x_i, self.clip_values[0], self.clip_values[1], x_i)
            x_i, _ = super().__call__(x_i)
            x_i = x_i - self.means
            x_i = x_i.transpose(np.argsort(self.transpose))
            x_i = x_i.astype(np.float32)
            x_new.append(x_i)
        return np.stack(x_new), y
