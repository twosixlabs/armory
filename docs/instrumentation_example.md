# Armory Instrumentation Examples: Capture Artifacts from Existing Code
For an introduction to `Probe`s and `Meter`s, please refer to [Measurement Overview](./metrics.md).

## Context
To monitor particular aspects of an `armory run` session, the user needs to know the following factors:
- What am I measuring?
- When should I measure it?
- Where should my custom monitoring script go?

We assume the user is capturing artifacts from the model or attack and wishes to use `Probe`s and `Meter`s to monitor certain variables within the code.

Recall the steps for a minimal working example (in [Measurement Overview](./metrics.md#instrumentation)):
1. Create `Probe` via `get_probe(name)`
2. Place `Probe` actions
3. Create `Meter` with processing functions that take input from created `Probe`
4. Connect `Meter` to `Hub` via `get_hub().connect_meter(meter)`

The examples will show how each of these steps are accomplished.

## Example 1: Measuring a Model Layer's Output
### User Story
I am interested in layer output from the second `relu` activation of a `forward` method located in `armory/baseline_models/pytorch/cifar.py`.
### `Probe` Example Code
The code below is an example of how to accomplish steps 1 and 2 (note the lines of code with `# added` comments at the end) for a model code that the user is modifying.
```python
"""
CNN model for 32x32x3 image classification
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier

from armory.instrument import get_probe # added
probe = get_probe("my_model") # added

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """
    This is a simple CNN for CIFAR-10 and does not achieve SotA performance
    """

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, 1)
        self.conv2 = nn.Conv2d(4, 10, 5, 1)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # from NHWC to NCHW
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x_out = x.detach().cpu().numpy() # added
        probe.update(layer_output=x_out) # added
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

...
```



#### Step 1
After importing `get_probe`, `probe = get_probe("my_model")` creates a `Probe` object with the name `"my_model"`, which is what the user can refer to later to apply processing functions through a `Meter` object.

#### Step 2
`x_out = x.detach().cpu().numpy()` is taking the layer output of interest (second `relu` activation output) and converting the tensor to `numpy` array on the CPU, which will be passed to `probe`. An updated value of `x_out` is stored in `layer_output` via `probe.update(layer_output=x_out)`. Like the `Probe` name `"my_model"`, `layer_output` can be referenced by the user later to apply additional processing functions through a `Meter` object.

### `Meter` Example Code
Now that a `Probe` instance has been created, we need to create a `Meter` object to accept any updated values from `Probe` and apply further processing that the user desires. Suppose the user created a script located at `armory/user_init.py` (Please refer to [User Initialization](./scenarios.md#user-initialization) for more details about using the `user_init` block):
```python
from armory.instrument import get_hub, Meter

def set_up_meter():
    meter = Meter(
        "my_arbitrary_meter_name", lambda x: x, "my_model.layer_output"
    )
    get_hub().connect_meter(meter)
```
#### Step 3
In this particular example, the `Meter` accepts 3 inputs: a meter name, a metric/function for processing, and a argument name to pass the metric/function.
- The meter name (`"my_arbitrary_meter_name"`) can be arbitrary within this context
- For the scope of this document, we only consider simple `Meter`s with the identity function as a metric i.e. `Meter` will record variables monitored by `Probe` as-is (thus `lambda x: x`)
- The argument passed to the metric/function follows a `.`-separated format (`"my_model.layer_output"`), which needs to be consistent with `Probe` setup earlier:
    - `my_model` matches input in `probe = get_probe("my_model")`
    - `layer_output` matches variable name in `probe.update(layer_output=x_out)`

#### Step 4
For the scope of this document, we don't dwell on what `armory` is doing with `get_hub().connect_meter(meter)` other than to mention this step is necessary for establishing the connection between `meter` created in `armory/user_init.py` and `probe` created in the modified version of `armory/baseline_models/pytorch/cifar.py`.

### Config Setup
Last but not least, the config file passed to `armory run` needs to be updated for these changes to take effect. Assuming the `"model"` block has been changed appropriately, the other block that needs to be added is `"user_init"` (please refer to [User Initialization](./scenarios.md#user-initialization) for more details about using the `user_init` block):
```json
...
    "model": {
        ...
    },
    "user_init": {
        "module": "user_init",
        "name": "set_up_meter"
    },
...
```
This will prompt armory to run `set_up_meter` in `user_init.py` before anything else is loaded for the scenario.

## Example 2: Measuring Attack Artifact
### User Story
I defined a custom attack with `CARLADapricotPatch` in `armory/custom_attack.py`, and I am interested in the patch after <ins>***every iteration***</ins>, which is generated by `CARLADapricotPatch._augment_images_with_patch` and returned as an output.
### `Probe` Example Code
```python
from armory.art_experimental.attacks.carla_obj_det_patch import CARLADapricotPatch
from armory.instrument import get_probe
probe = get_probe("my_attack")

class CustomAttack(CARLADapricotPatch):
    def _augment_images_with_patch(self, **kwargs):
        return_value = super()._augment_images_with_patch(**kwargs)
        x_patch, patch_target, transformations = return_value
        probe.update(attack_output=x_patch)

        return return_value
```
#### Step 1
This step is the same as before, except `Probe` name is set to`"my_attack"`, which is what the user can refer to later to apply processing functions through a `Meter` object.

#### Step 2
The only difference between `CustomAttack` and `CARLADapricotPatch` is that `_augment_images_with_patch` has been redefined to call on `CARLADapricotPatch._augment_images_with_patch` and then have `probe` update the value for `x_patch` that results from that call. An updated value of `x_patch` is stored in `attack_output` via `probe.update(attack_output=x_patch)`. Like the `Probe` name `"my_attack"`, `attack_output` can be referenced by the user later to apply additional processing functions through a `Meter` object.

### `Meter` Example Code
Now that a `Probe` instance has been created, we need to create a `Meter` object to accept any updated values from `Probe` and apply further processing that the user desires. Suppose the user created a script located at `armory/user_init.py` (Please refer to [User Initialization](./scenarios.md#user-initialization) for more details about using the `user_init` block):
```python
from armory.instrument import get_hub, Meter

def set_up_meter():
    meter = Meter(
        "my_arbitrary_meter_name", lambda x: x, "my_attack.attack_output"
    )
    get_hub().connect_meter(meter)
```
#### Step 3
As before, the `Meter` accepts 3 inputs: a meter name, a metric/function for processing, and a argument name to pass the metric/function.
- The meter name (`"my_arbitrary_meter_name"`) can be arbitrary within this context
- Again, `Meter` will record variables monitored by `Probe` as-is (thus `lambda x: x`)
- The argument passed to the metric/function follows a `.`-separated format (`"my_attack.attack_output"`), which needs to be consistent with `Probe` setup earlier:
    - `my_attack` matches input in `probe = get_probe("my_attack")`
    - `attack_output` matches variable name in `probe.update(attack_output=x_patch)`

#### Step 4
Again, `get_hub().connect_meter(meter)` is necessary for establishing the connection between `meter` created in `armory/user_init.py` and `probe` created in `armory/custom_attack.py`.

### Config Setup
Last but not least, the config file passed to `armory run` needs to be updated for these changes to take effect. Assuming the `"attack"` block has been changed appropriately, the other block that needs to be added is `"user_init"` (please refer to [User Initialization](./scenarios.md#user-initialization) for more details about using the `user_init` block):
```json
...
    "attack": {
        ...
    },
    "user_init": {
        "module": "user_init",
        "name": "set_up_meter"
    },
...
```
This will prompt armory to run `set_up_meter` in `user_init.py` before anything else is loaded for the scenario.

## Saving Results
By default, outputs from `Meter`s will be saved to the output `json` file after `armory run`. Whether this suffices for the user depends on what the user is trying to measure.

Users who have tried the examples in this document, however, may run into some of the following warning logs:
> 2022-12-16 19:34:36 30s WARNING  armory.instrument.instrument:_write:856 record (name=my_arbitrary_meter_name, batch=0, result=...) size > max_record_size 1048576. Dropping.

This is because of `Meter`'s default settings, which has a size limit for each record. That the outputs exceed a size limit also suggests that a `json` file may not be the best file type to save to. To override these behaviors, we need a new `Meter` subclass to work with our examples that does not have a size limit and will save to another filetype such as a `pkl` file. Below is an updated `user_init.py` for Example 2 with a new `PickleMeter` class:
```python
from armory.instrument import get_hub, Meter
import pickle

class PickleMeter(Meter):
    def __init__(
        self,
        name,
        output_dir,
        x,
        max_batches=None,
    ):
        """
        :param name (string): name given to PickleMeter
        :param output_dir (string): 
        :param x (string): .-separated string of probe name and variable name e.g. "scenario.x"
        :param max_batches (int or None): maximum number of batches to export
        """
        metric_args = [x]
        super().__init__(name, lambda x: x, *metric_args)

        self.max_batches = max_batches
        self.metric_args = metric_args
        self.output_dir = output_dir
        self.iter = 0

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        batch_num, batch_data = self.arg_batch_indices[0], self.values[0]
        if self.max_batches is not None and batch_num >= self.max_batches:
            return

        with open(os.path.join(self.output_dir, f"{self.name}_batch_{batch_num}_iter_{self.iter}.pkl"), "wb") as f:
            pickle.dump(batch_data, f)
        self.iter += 1
        if clear_values:
            self.clear()
        self.never_measured = False

def set_up_meter():
    meter = PickleMeter(
        "my_arbitrary_meter_name", get_hub().export_dir, "my_attack.attack_output"
    )
    get_hub().connect_meter(meter, use_default_writers=False)
```