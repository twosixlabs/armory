# Introduction
As a first step, we will demonstrate how to implement a custom attack - one which does not fit directly into the existing attack types supported by ARMORY.

# Goal
Starting with the defended CIFAR10 scenario [here](../../scenario_configs/eval1-4/cifar/cifar10_baseline.json), we want to modify the attack as follows: rather than a simple untargeted attack, launch a targeted projected gradient descent (PGD) attack against each incorrect class, one at a time, until a successful adversarial example is identified.

# Implementation
Because ARMORY targeted attacks are designed to attack only one class at a time, and not repeat the attack with a different target for the same example, this requires a custom attack.

First, we will modify our scenario file to point to a custom attack class.  The attack that is loaded is controlled by the `module` and `name` fields.  We will implement our attack in a class named `CustomAttack` which will live in the `custom_attack.py` file in this directory.  The various parameters in `kwargs` align with the arguments of the `ProjectedGradientDescent` class in ART, and will control how each of these attacks is performed.  A word of caution: despite wanting to implement a targeted attack at each stage of the custom attack, we will set the `targeted` field to `false`.  This is because a standard targeted PGD attack will attack only a single class and the attack would receive a target label as determined by ARMORY or as further configured in the scenario file.  Instead, we will set the `use_label` field to `true` because our attack requires knowledge of the ground truth class in order to cycle through incorrect target classes.  The updated attack configuration is shown below:

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
    "module": "custom_attack",
    "name": "CustomAttack",
    "use_label": true
},
```

Next, we will create a new class for the custom attack.  Because the attack will consist of repeated applications of a PGD attack, our new class will inherit from ART's `ProjectedGradientDescent` class.  Our new class will override the `__init__` method of the parent class, to control attack configuration, and the `generate` method to control how the attack actually functions.

First, we implement the `__init__` method, which receives as arguments the estimator (i.e. classifier) and keyword arguments (kwargs).  Because the PGD arguments are configured in the scenario file, the only change required here is to configure the attack as targeted.  By configuring the attack as a targeted one here, but not in the scenario file, we bypass a check that both `targeted` and `use_label` are set to `true`, and allow our attack to receive ground truth labels while still using a targeted attack.  We then initialize the PGD attack by calling the `__init__` method of the parent class.  The complete `__init__` method is shown below

```python
import numpy as np

from art.attacks.evasion import ProjectedGradientDescent

class CustomAttack(ProjectedGradientDescent):
    def __init__(self, estimator, **kwargs):
        modified_kwargs = kwargs.copy()
        modified_kwargs["targeted"] = True
        super().__init__(estimator, **modified_kwargs)
```

Next, we write the `generate` method where the actual attack logic will be implemented.  Our attack will receive two arguments: a set of benign images `x` and the ground truth labels `y`.  Our attack will iterate over a single image in the batch at a time.  For each image, our attack should iterate over all possible classes (i.e. classes 0-9), perform a targeted attack against each class except the ground truth class, and quit once a successful adversarial example is found.  Since our attack inherits from `ProjectedGradientDescent`, we can call the `generate` method of the parent class to implement each PGD attack with the target label.  To check whether the generated example is successful, we can use the `self.estimator.predict` method to determine the predicted classes for each generated image.  The full `generate` method of our custom class is shown below.

```python
def generate(self, x, y):

    x_adv = []
    for x_sample, y_sample in zip(x, y):
        for target in range(10):

            # Do not target correct class
            if target == y_sample:
                continue

            # Generate sample targeting `target` class
            y_target = np.zeros((1, 10), dtype=np.int64)
            y_target[0, target] = 1
            x_adv_sample = super().generate(
                np.expand_dims(x_sample, axis=0), y_target
            )

            # Check - does this example fool the classifier?
            x_adv_pred = np.argmax(self.estimator.predict(x_adv_sample))
            if x_adv_pred != y_sample:
                break
        x_adv.append(x_adv_sample)

    x_adv = np.concatenate(x_adv, axis=0)
    return x_adv
```

Finally, so that the custom model and attack code are discovered regardless of which directory ARMORY is run from, we update the `local_repo_path` field in the configuration file to point to the directory containing this code.

```json
"local_repo_path": "twosixlabs/armory/tutorials/adaptive_attacks",
```

# Complete Example
The complete example is demonstrated via the following files:
* [Configuration File](./custom_attack.json)
* [Custom Attack](./custom_attack.py)

This example may be run with the following command:
```
armory run tutorials/adaptive_attacks/custom_attack.json
```
