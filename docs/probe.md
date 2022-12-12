# Probes and Meters: Advanced End-to-End Examples
For an introduction to `Probe`s and `Meter`s, please refer to [Measurement Overview](./metrics.md).

## Context
To monitor particular aspects of an `armory run` session, the user needs to know the following factors:
- What am I measuring?
- When should I measure it?
- Where should my custom monitoring script go?

The examples in this section highlight the nuances of using `Probe`s and `Meter`s for flexible monitoring arrangements in `armory`.

## Example 1: Model Layer Output
### User Story
I have a `PyTorchFasterRCNN` model and I am interested in output from the `relu` activation of the third (index 2) `Bottleneck` of `layer4`
### Example Code
This is an example of working with a python package/framework (i.e. `pytorch`) that comes with built-in hooking mechanisms. In the code snippet below, we are relying on an existing function `register_forward_hook` to monitor the layer of interest:
```python showLineNumbers
from armory.scenarios.main import get as get_scenario
from armory.instrument import get_probe, Meter, get_hub

# load Scenario
s = get_scenario(
    "/armory/tmp/2022-11-03T180812.020999/carla_obj_det_adversarialpatch_undefended.json",
    num_eval_batches=1,
).load()

# create Probe with "test" namespace
probe = get_probe("test")

# define the hook to pass to "register_forward_hook"
# the signature of 3 inputs is what pytorch expects
# hook_module refers to the layer of interest, but is not explicitly referenced when passing to register_forward_hook
def hook_fn(hook_module, hook_input, hook_output): 
    probe.update(lambda x: x.detach().cpu().numpy(), layer4_2_relu=hook_output[0][0]) # [0][0] for slicing

# register hook
# the hook_module mentioned earlier is referenced via s.model.model.backbone.body.layer4[2].relu
# the register_forward_hook method call must be passing self as a hook_module to hook_fn
s.model.model.backbone.body.layer4[2].relu.register_forward_hook(hook_fn)

# create Meter for Probe with "test" namespace
meter = Meter("layer4_2_relu", lambda x: x, "test.layer4_2_relu")

# connect Meter to Hub
get_hub().connect_meter(meter)

s.next()
s.run_attack()
```

### Packages with Hooks
That a package provides a hooking mechanism is convenient, but the user also has to be aware of the what to pass to the hooking mechanism as well as what format to pass it in. Please reference [`pytorch` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) for more details regarding this example.

Note that `pytorch` also provides other hooking functionality such as:
- `register_forward_pre_hook`
- `register_full_backward_hook`

### Probe and Meter Details
Aside the specifics of using `register_forward_hook`, consider how `Probe` and `Meter` are incorporated in this example. Recall the 4 steps for a minimal working example (in [Measurement Overview](./metrics.md)):
1. Create `Probe` via `get_probe("test")`
2. Define `Probe` actions
3. Create `Meter` with processing functions that take input from created `Probe`
4. Connect `Meter` to `Hub` via `get_hub().connect_meter(meter)`

#### Step 1
Note the input `"test"` that is passed in `get_probe("test")` - this needs to match with the first portion of a `.`-separated argument name `"test.layer4_2_relu"` that is passed to creating a `Meter` in [Step 3](#step3)

#### Step 2
The `update` method for `Probe` takes as input optional processing functions and variable names and corresponding values that are to be monitored.
- The variable name `layer4_2_relu` is how we are choosing to reference a certain value
    - this needs to match with the second portion of a `.`-separated argument name `"test.layer4_2_relu"` that is passed to creating a `Meter` in [Step 3](#step3)
- `hook_output[0][0]` is the value we are interested in, which is the output from `s.model.model.backbone.body.layer4[2].relu` after a forward pass
    - `[0][0]` was included to slice the output to show that it can be done, and because we know the shape of the output in advance
- `lambda x: x.detach().cpu().numpy()` is the processing function that converts `hook_output[0][0]` from a tensor to an array

#### Step 3<a name="step3"></a>
In this particular example, the `Meter` accepts 3 inputs: a meter name, a metric/function for processing, and a argument name to pass the metric/function.
- The meter name (`"layer4_2_relu"`) can be arbitrary within this context
- For the scope of this section, we only consider simple `Meter`s with the identity function as a metric i.e. `Meter` will record variables monitored by `Probe` as-is (thus `lambda x: x`)
- The argument passed to the metric/function follows a `.`-separated format (`"test.layer4_2_relu"`), which needs to be consistent with `Probe` setup:
    - `test` matches input in `get_probe("test")`
    - `layer4_2_relu` matches variable name in `layer4_2_relu=hook_output[0][0]`

#### Step 4
 We don't dwell on what `armory` is doing in step 4 with `get_hub().connect_meter(meter)` other than to mention this step is necessary.