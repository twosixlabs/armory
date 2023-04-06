# Armory Instrumentation Examples: Measuring Experiment Artifacts Using Probes and Meters
For an introduction to `Probe`s and `Meter`s, please refer to [Measurement Overview](./metrics.md#instrumentation). We assume the user is capturing artifacts from the model or attack and wishes to use `Probe`s and `Meter`s to monitor certain variables within the code.

Recall the steps for a minimal working example (in [Measurement Overview](./metrics.md#instrumentation)):
1. Create `Probe` via `get_probe(name)`
2. Place `Probe` actions
3. Create `Meter` with processing functions that take input from created `Probe`
4. Connect `Meter` to `Hub` via `get_hub().connect_meter(meter)`

The examples will show how each of these steps are accomplished.

## Example 1: Measuring a Model Layer's Output
### User Story
I am interested in the layer output from the second `relu` activation of a `forward` method located in `armory/baseline_models/pytorch/cifar.py`.
### `Probe` Example Code
The code below is an example of how to accomplish steps 1 and 2 (note the lines of code with `# added` comments at the end) for a model code that the user is modifying.
```python
"""
CNN model for 32x32x3 image classification
"""
...

from armory.instrument import get_probe # added
probe = get_probe("my_model") # added

class Net(nn.Module):
    ...

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
Now that a `Probe` instance has been created, we need to create a `Meter` object to accept any updated values from `Probe` and apply further processing that the user desires. We can create the `Meter` in a function added to a local Python script we'll name `user_init_script.py`. In [Config Setup](#config-setup) shortly below, we'll show how to ensure this code is run during scenario initialization.
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
For the scope of this document, we don't dwell on what `armory` is doing with `get_hub().connect_meter(meter)` other than to mention this step is necessary for establishing the connection between `meter` created in `armory/user_init_script.py` and `probe` created in the modified version of `armory/baseline_models/pytorch/cifar.py`.

### Config Setup
Last but not least, the config file passed to `armory run` needs to be updated for these changes to take effect, which is accomplished by adding the `"user_init"` block (please refer to [User Initialization](./scenarios.md#user-initialization) for more details):
```json
...
    "user_init": {
        "module": "user_init_script",
        "name": "set_up_meter"
    },
...
```
This will prompt armory to run `set_up_meter` in `user_init_script.py` before anything else is loaded for the scenario.

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
As in [Example 1](#meter-example-code), we need to create a `Meter` object to accept any updated values from `Probe` and apply further processing that the user desires. We can create the `Meter` in a function added to a local Python script `user_init_script.py`. In [Config Setup](#config-setup-1) shortly below, we'll show how to ensure this code is run during scenario initialization.
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
Again, `get_hub().connect_meter(meter)` is necessary for establishing the connection between `meter` created in `armory/user_init_script.py` and `probe` created in `armory/custom_attack.py`.

### Config Setup
Last but not least, the config file passed to `armory run` needs to be updated for these changes to take effect, which is accomplished by adding the `"user_init"` block (please refer to [User Initialization](./scenarios.md#user-initialization) for more details):
```json
...
    "user_init": {
        "module": "user_init_script",
        "name": "set_up_meter"
    },
...
```
This will prompt armory to run `set_up_meter` in `user_init_script.py` before anything else is loaded for the scenario.

## Saving Results
By default, outputs from `Meter`s will be saved to the output `json` file after `armory run`. Whether this suffices for the user depends on what the user is trying to measure.

Users who have tried the examples in this document, however, may run into some of the following warning logs:
> 2022-12-16 19:34:36 30s WARNING  armory.instrument.instrument:_write:856 record (name=my_arbitrary_meter_name, batch=0, result=...) size > max_record_size 1048576. Dropping.

Outputs are saved to a `json` file because of a default `ResultWriter` class tied to the `Meter` class, which has a `max_record_size` limit for each record. Any record that exceeds `max_record_size` will not save to the `json` file. That the outputs exceed a size limit also suggests that a `json` file may not be the best file type to save to. To work around these behaviors, we can define a new `Writer` subclass (`ResultWriter` is also a `Writer` subclass) to work with our examples that does not have a size limit and will save to another filetype, such as a `png` file, since we are saving data for an image. Below is an updated `user_init_script.py` for Example 2 with a new `ImageWriter` class, which uses the `export` method of `ObjectDetectionExporter` to save an image, and a `set_up_meter_writer` function that will be executed with the `user_init` block:
```python
from armory.instrument import get_hub, Meter, Writer
from armory.instrument.export import ObjectDetectionExporter

class ImageWriter(Writer):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.iter_step = 0
        self.current_batch_index = 0
        self.exporter = ObjectDetectionExporter(self.output_dir)

    def _write(self, name, batch, result):
        if batch != self.current_batch_index:
            self.current_batch_index = batch # we are on a new batch
            self.iter_step = 0               # restart iter_step count
        basename = f"{name}_batch_{batch}_iter_{self.iter_step}"
        # assume single image per batch: result[0]
        self.exporter.export(x = result[0], basename = basename)
        self.iter_step += 1 # increment iter_step

def set_up_meter_writer():
    meter = Meter(
        "my_attack_identity", lambda x: x, "my_attack.attack_output"
    )
    writer = ImageWriter(output_dir = get_hub().export_dir)
    meter.add_writer(writer)
    get_hub().connect_meter(meter, use_default_writers=False)
```