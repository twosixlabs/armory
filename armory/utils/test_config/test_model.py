import warnings
import numpy as np

from armory.utils.config_loading import load_model
from armory.utils.configuration import load_config
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.speech_recognition import SpeechRecognizerMixin
from art.estimators.object_detection import ObjectDetectorMixin


class TestModel:
    def _get_model(self, scenario_file):
        config = load_config(scenario_file)
        model, _ = load_model(config["model"])
        if not isinstance(
            model, (ClassifierMixin, SpeechRecognizerMixin, ObjectDetectorMixin)
        ):
            raise TypeError(f"Unsupported model type: {type(model)}")
        return model

    def _get_input_shape(self, model):
        if isinstance(model, ClassifierMixin):
            if hasattr(model, "input_shape"):
                input_shape = np.array(model.input_shape)
                if np.all(input_shape == (None,)):
                    input_shape = np.array((10000,))
                    warnings.warn("No model input shape specified. Assuming (10000,)")
                elif np.all(input_shape == (None, None, 3)):
                    input_shape = np.array((32, 32, 3))
                    warnings.warn(
                        "No model input shape specified. Assuming (32, 32, 3,)"
                    )
            else:
                warnings.warn("No model input shape specified. Assuming (32, 32, 3)")
                input_shape = np.array((32, 32, 3))
        elif isinstance(model, ObjectDetectorMixin):
            if hasattr(model, "input_shape"):
                warnings.warn("No model input shape specified. Assuming (32, 32, 3)")
            input_shape = np.array((32, 32, 3))
        elif isinstance(model, SpeechRecognizerMixin):
            if hasattr(model, "input_shape"):
                warnings.warn("No model input shape specified. Assuming (32, 32, 3)")
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
        if isinstance(model, ClassifierMixin) or isinstance(model, ObjectDetectorMixin):
            test_ground_truth = test_output
            if (
                isinstance(test_output, np.ndarray)
                and test_output.ndim == 2
                and test_output.shape[1] > 1
            ):
                test_ground_truth = np.zeros((1, model.nb_classes))
                test_ground_truth[:, 0] = 1
        elif isinstance(model, SpeechRecognizerMixin):
            test_ground_truth = np.array(["HELLO"])
        return test_ground_truth

    def test_model(self, scenario_file):
        model = self._get_model(scenario_file)
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
        if hasattr(model, "nb_classes"):
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

        except:
            warnings.warn(
                "Model encountered error during gradient computation. White-box evaluation may be limited"
            )

        # verify gradient output shape, and did not overwrite inputs
        if test_grad is not None:
            assert np.all(
                test_grad.shape == test_input.shape
            ), f"For input of size {test_input.shape} got gradient of size {test_grad.shape}"
            assert np.all(
                test_input == copy_input
            ), "The gradient computation overwrites the input x value"
            assert np.all(
                test_ground_truth == copy_ground_truth
            ), "The gradient computation overwrites the input y value"

        # verify gradient contains no NaN or Inf values
        if test_grad is not None:
            if np.any(np.isnan(test_grad)) or np.any(np.isinf(test_grad)):
                warnings.warn("NaN/Inf values detected in model gradient")
            if np.all(test_grad == 0):
                warnings.warn("All-zero gradients detected")
