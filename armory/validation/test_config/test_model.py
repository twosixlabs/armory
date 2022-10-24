import json
import warnings

from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.object_detection import ObjectDetectorMixin
from art.estimators.speech_recognition import SpeechRecognizerMixin
import numpy as np

from armory.utils.config_loading import load_model


class TestModel:
    def _get_model(self, model_config):
        model, _ = load_model(model_config)
        if not isinstance(
            model, (ClassifierMixin, SpeechRecognizerMixin, ObjectDetectorMixin)
        ):
            raise TypeError(f"Unsupported model type: {type(model)}")
        return model

    def _get_input_shape(self, model):
        if isinstance(model, ClassifierMixin):
            if hasattr(model, "input_shape"):
                input_shape = np.array(model.input_shape)
            else:
                warnings.warn("No model input shape specified. Assuming (32, 32, 3)")
                input_shape = np.array((32, 32, 3))
            if np.all(input_shape == (None,)):
                warnings.warn("Model shape given as (None,), assuming (10000,)")
                input_shape = np.array((10000,))
            elif None in input_shape:
                test_input_shape = input_shape.copy()
                test_input_shape[np.equal(test_input_shape, None)] = 32
                warnings.warn(
                    f"Model shape given as {input_shape}. Assuming {test_input_shape}"
                )
                input_shape = test_input_shape
        elif isinstance(model, ObjectDetectorMixin):
            input_shape = np.array((32, 32, 3))
            warnings.warn(
                "Object detector model does not specify input shape. Assuming (32, 32, 3)"
            )
        elif isinstance(model, SpeechRecognizerMixin):
            warnings.warn(
                "Speech recognition model does not specify input shape. Assuming (10000,)"
            )
            input_shape = np.array((10000,))
        return input_shape

    def _get_clip_values(self, model):
        if isinstance(model, ClassifierMixin) or isinstance(model, ObjectDetectorMixin):
            assert (
                model.clip_values is not None
            ), "Clip values not provided. Clip values are required to keep input values within valid range"
            clip_min, clip_max = model.clip_values
        elif isinstance(model, SpeechRecognizerMixin):
            clip_min, clip_max = (-1.0, 1.0)
        return clip_min, clip_max

    def _get_test_input(self, input_shape, clip_min, clip_max):
        test_sample = np.random.randn(1, *input_shape).astype(np.float32)
        test_sample = np.clip(test_sample, clip_min, clip_max)
        return test_sample

    def _get_test_ground_truth(self, model, test_output):
        if isinstance(model, ClassifierMixin):
            test_ground_truth = test_output
            if (
                isinstance(test_output, np.ndarray)
                and test_output.ndim == 2
                and test_output.shape[1] > 1
            ):
                test_ground_truth = np.zeros((1, model.nb_classes))
                test_ground_truth[:, np.argmin(test_output, axis=1)[0]] = 1
        elif isinstance(model, ObjectDetectorMixin):
            test_ground_truth = [
                {
                    "boxes": np.reshape(np.arange(0.0, 1.0, 1.0 / 32.0), (8, 4)),
                    "labels": np.ones((8,), dtype=np.int64),
                    "scores": np.ones((8,), dtype=np.float32),
                }
            ]
        elif isinstance(model, SpeechRecognizerMixin):
            test_ground_truth = np.array(["HELLO"])
        return test_ground_truth

    def test_model(self, model_config):

        # verify model loads without error
        model_config = json.loads(model_config)
        model = self._get_model(model_config)
        input_shape = self._get_input_shape(model)

        # verify model clip values broadcast to input shape
        clip_min, clip_max = self._get_clip_values(model)
        np.broadcast_to(clip_min, input_shape)
        np.broadcast_to(clip_max, input_shape)

        # verify model forward pass
        test_input = self._get_test_input(input_shape, clip_min, clip_max)
        copy_input = test_input.copy()
        test_output = model.predict(test_input)
        assert np.all(
            test_input == copy_input
        ), "Model prediction overwrites the input x value"
        if isinstance(model, ClassifierMixin) and hasattr(model, "nb_classes"):
            assert (
                test_output.shape[1] == model.nb_classes
            ), f"Model configured for {model.nb_classes} output classes, but output shape is {test_output.shape}"

        # test model gradient
        test_ground_truth = self._get_test_ground_truth(model, test_output)
        copy_ground_truth = test_ground_truth.copy()
        test_grad = None
        try:
            test_grad = model.loss_gradient(test_input, test_ground_truth)
            if test_grad is None:
                warnings.warn(
                    "Model returned None gradient. White-box evaluation may be limited"
                )
            else:
                test_grad = np.stack(test_grad)

        except Exception:
            warnings.warn(
                "Model encountered error during gradient computation. White-box evaluation may be limited"
            )

        # test returned gradients for issues
        if test_grad is not None:
            if not np.all(test_grad.shape == test_input.shape):
                warnings.warn(
                    f"For input of size {test_input.shape} got gradient of size {test_grad.shape}"
                )
            if np.any(np.isnan(test_grad)) or np.any(np.isinf(test_grad)):
                warnings.warn("NaN/Inf values detected in model gradient")
            if np.all(test_grad == 0):
                warnings.warn("All-zero gradients detected")
            assert np.all(
                test_input == copy_input
            ), "The gradient computation overwrites the input x value"
            assert np.all(
                test_ground_truth == copy_ground_truth
            ), "The gradient computation overwrites the input y value"
