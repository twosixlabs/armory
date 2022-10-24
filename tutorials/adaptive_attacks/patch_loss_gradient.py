from art.attacks.evasion import ProjectedGradientDescent
from patch_loss_gradient_model import get_art_model
import torch
from torch.autograd import Variable
from torchvision.transforms import RandomErasing

from armory.utils.evaluation import patch_method

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomAttack(ProjectedGradientDescent):
    def __init__(self, estimator, **kwargs):

        # Create copy of the model (to avoid overwriting loss_gradient of original model)
        new_estimator = get_art_model(model_kwargs={}, wrapper_kwargs={})
        new_estimator.model.load_state_dict(estimator.model.state_dict())
        # OR:
        # import copy
        # new_estimator = copy.deepcopy(estimator)

        # Point attack to copy of model
        super().__init__(new_estimator, **kwargs)

        @patch_method(new_estimator)
        def loss_gradient(
            self, x: "torch.Tensor", y: "torch.Tensor", **kwargs
        ) -> "torch.Tensor":
            x_var = Variable(x, requires_grad=True)
            y_cat = torch.argmax(y)

            transform = RandomErasing(p=1.0, scale=(0.5, 0.5))
            x_mod = torch.stack([transform(x_var[0]) for i in range(100)], dim=0)
            logits = self.model.net.forward(x_mod)
            loss = self._loss(logits, y_cat.repeat(100))

            self._model.zero_grad()
            loss.backward()
            grads = x_var.grad
            return grads
