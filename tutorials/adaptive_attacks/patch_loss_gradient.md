# Introduction
Here, we demonstrate how to patch the `loss_gradient` method of the model, allowing for control over how the gradient is computed for white box attacks.

# Goal
Starting with the baseline CIFAR10 scenario [here](../../scenario_configs/eval1-4/cifar/cifar10_baseline.json), we will slightly modify the model to apply a random transformation to each input image before classification.  In particular, we will apply the `RandomErasing` operation, implemented in Torch, to each input image, then run each modified image through a forward pass of the model.

Because this operation is random, one might want to average the gradient over many instances of the random transformation as part of a white box attack.  Here, we demonstrate how to achieve this via modifying the `loss_gradient` method of the model.

# Implementation
First, we modify the model to apply a `RandomErasing` operation to each input before classification.  The modified model is available [here](./patch_loss_gradient_model.py).  We also update the scenario configuration file to point to the updated model.  The relevant lines of the updated configuration are shown below.

```json
"model": {
    "fit": true,
    "fit_kwargs": {
        "nb_epochs": 20
    },
    "model_kwargs": {},
    "module": "patch_loss_gradient_model",
    "name": "get_art_model",
    "weights_file": null,
    "wrapper_kwargs": {}
},
```

Next, we will follow the procedure outlined in [Tutorial 1](./custom_attack.md) to create a `CustomAttack` class, which will give us flexibility over how the attack is performed.  This is accomplished via changes to the `module` and `name` parameters of the attack to point to a custom class.  The updated attack configuration is shown below.

```json
"attack": {
    "knowledge": "white",
    "kwargs": {
        "batch_size": 1,
        "eps": 0.031,
        "eps_step": 0.007,
        "max_iter": 20,
        "num_random_init": 1,
        "random_eps": false,
        "targeted": false,
        "verbose": false
    },
    "module": "patch_loss_gradient",
    "name": "CustomAttack",
    "use_label": true
},
```

Again, our custom class will inherit from the `ProjectGradientDescent` class defined in ART, and requires updates to the `__init__` method.  First, we update the `__init__` method to initialize a PGD attack with a *copy* of the trained model.  Copying the model is necessary so that updates to the `loss_gradient` method do not impact the original model under evaluation, but only the model used to compute gradients for the attack.  If saved weights for the model are available, a copy of the model may be achieved by passing the model weight file to `get_art_model`.  In this case, since the model is trained as part of the scenario, the weights may be copied to the new model via the `load_state_dict` method.  These updates are seen in the code snippet below.

```python
class CustomAttack(ProjectedGradientDescent):
    def __init__(self, estimator, **kwargs):
        
        # Create copy of the model (to avoid overwriting loss_gradient of original model)
        new_estimator = get_art_model(model_kwargs={}, wrapper_kwargs={})
        new_estimator.model.load_state_dict(estimator.model.state_dict())

        # Point attack to copy of model
        super().__init__(new_estimator, **kwargs)
```

Then, we use the `patch_method` decorator from ARMORY to define a new `loss_gradient` method that impacts only the copy of the classifier.  In this case, for each image passed to the model, we compute 100 random transformations, run a forward pass of the model on each transformed version of the image, and average the gradient across all 100 instances.  The patched `loss_gradient` code is shown below.

```python
@patch_method(new_estimator)
def loss_gradient(self, x: "torch.Tensor", y: "torch.Tensor", **kwargs) -> "torch.Tensor":

    x_var = Variable(x, requires_grad=True)
    y_cat = torch.argmax(y)
    
    transform = RandomErasing(p=1., scale=(0.5, 0.5))
    x_mod = torch.stack([transform(x_var[0]) for i in range(100)], dim=0)
    logits = self.model.net.forward(x_mod)
    loss = self._loss(logits, y_cat.repeat(100))

    self._model.zero_grad()
    loss.backward()
    grads = x_var.grad
    return grads
```

# Complete Example
The complete example is demonstrated via the following files:
* [Configuration File](./patch_loss_gradient.json)
* [Model Under Evaluation](./patch_loss_gradient_model.py)
* [Custom Attack with Patched Gradient](./patch_loss_gradient.py)

This example may be run with the following command:
```
armory run tutorials/adaptive_attacks/patch_loss_gradient.json
```
