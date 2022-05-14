import numpy as np

class ImageContext:
    def __init__(self, x_shape):
        self.x_shape = x_shape
        self.input_type = np.uint8
        self.input_min = 0
        self.input_max = 255

        self.quantization = 255

        self.output_type = np.float32
        self.output_min = 0.0
        self.output_max = 1.0


#
# mnist_context = ImageContext(x_shape=(28, 28, 1))

def check_shapes(actual, target):
    """
    Ensure that shapes match, ignoring None values

    actual and target should be tuples
        actual should not have None values
    """
    if type(actual) != tuple:
        raise ValueError(f"actual shape {actual} is not a tuple")
    if type(target) != tuple:
        raise ValueError(f"target shape {target} is not a tuple")
    if None in actual:
        raise ValueError(f"None should not be in actual shape {actual}")
    if len(actual) != len(target):
        raise ValueError(f"len(actual) {len(actual)} != len(target) {len(target)}")
    for a, t in zip(actual, target):
        if a != t and t is not None:
            raise ValueError(f"shape {actual} does not match shape {target}")


def preprocessing_chain(*args):
    """
    Wraps and returns a sequence of functions
    """
    functions = [x for x in args if x is not None]
    if not functions:
        return None

    def wrapped(x):
        for function in functions:
            x = function(x)
        return x

    return wrapped

#
# def mnist_canonical_preprocessing(batch):
#     return canonical_image_preprocess(mnist_context, batch)
#

# def canonical_image_preprocess(context, batch):
#     check_shapes(batch.shape, (None,) + context.x_shape)
#     if batch.dtype != context.input_type:
#         raise ValueError("input batch dtype {batch.dtype} != {context.input_type}")
#     assert batch.min() >= context.input_min
#     assert batch.max() <= context.input_max
#
#     batch = batch.astype(context.output_type) / context.quantization
#
#     if batch.dtype != context.output_type:
#         raise ValueError("output batch dtype {batch.dtype} != {context.output_type}")
#     assert batch.min() >= context.output_min
#     assert batch.max() <= context.output_max
#
#     return batch

class CanonicalImagePreprocessor(object):

    def __init__(self, context):
        self.context = context

    def check(self, batch):
        check_shapes(batch.shape, (None,) + self.context.x_shape)
        if batch.dtype != self.context.input_type:
            raise ValueError(f"input batch dtype: {batch.dtype} != context input type: {self.context.input_type}")
        assert batch.min() >= self.context.input_min
        assert batch.max() <= self.context.input_max

        batch = batch.astype(self.context.output_type) / self.context.quantization

        if batch.dtype != self.context.output_type:
            raise ValueError(f"output batch dtype {batch.dtype} != context {self.context.output_type}")
        assert batch.min() >= self.context.output_min
        assert batch.max() <= self.context.output_max
        return batch

