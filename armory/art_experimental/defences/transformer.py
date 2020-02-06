import copy

from art.classifiers import PyTorchClassifier
from torch import nn
from torch.nn.modules import activation

from armory.art_experimental.defences import base


class ReluX(activation.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReluX}(x) = \min(\max(0,x), X)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::
        >>> m = ReluX(1.5)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, max_value, inplace=False):
        max_value = float(max_value)
        if max_value <= 0.0:
            raise ValueError(f"max_value must be > 0, not {max_value}")
        super().__init__(0.0, max_value, inplace=inplace)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class ReluToReluXInner(base.Transformer):
    def __init__(self, in_place=False, max_value=6.0):
        """
        in_place - whether to modify the neural network in place or copy
        max_value - positive float to clip activation values to
        """
        self.in_place = bool(in_place)
        self.max_value = float(max_value)

    def transform(self, model):
        if isinstance(model, nn.Module):
            if not self.in_place:
                model = copy.deepcopy(model)
            self._torch_recursive_replace(model)
        else:
            raise NotImplementedError("model is not a torch module")
        return model

    def _torch_recursive_replace(self, module):
        for key, value in module._modules.items():
            if key == "relu" and isinstance(value, activation.ReLU):
                module._modules[key] = ReluX(
                    inplace=value.inplace, max_value=self.max_value
                )
            else:
                self._torch_recursive_replace(value)


class ReluToReluX(ReluToReluXInner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, classifier):
        if not isinstance(classifier, PyTorchClassifier):
            raise ValueError("classifier is not a PyTorchClassifier")
        # double ._model is needed due to ART's creation of an extra torch wrapper
        model = super().transform(classifier._model._model)
        return PyTorchClassifier(
            model,
            loss=classifier._loss,
            optimizer=classifier._optimizer,
            input_shape=classifier._input_shape,
            nb_classes=classifier._nb_classes,
            preprocessing=None,
        )
