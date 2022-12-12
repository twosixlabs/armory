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
I have a `PyTorchFasterRCNN` model and I am interested in output from the `relu` activation of the second `Bottleneck` of `layer4`
### Example Code
This is an example of working with a python packages/framework (i.e. `pytorch`) that comes with built-in hooking mechanisms. In the code snippet below, we are relying on an existing function `register_forward_hook` to monitor the layer of interest:
```python
from armory.scenarios.main import get as get_scenario

s = get_scenario(
    "/armory/tmp/2022-11-03T180812.020999/carla_obj_det_adversarialpatch_undefended.json",
    num_eval_batches=1,
).load()

# create the probe with "test" namespace
probe = get_probe("test")

# define the hook to pass to "register_forward_hook"
def hook_fn(hook_module, hook_input, hook_output):
    probe.update(lambda x: x.detach().cpu().numpy(), layer4_2_relu=hook_output[0][0]) # [0][0] for slicing

s.model.model.backbone.body.layer4[2].relu.register_forward_hook(hook_fn)

meter = Meter("layer4_2_relu", lambda x: x[0,0,:,:], "test.layer4_2_relu")
get_hub().connect_meter(meter)

s.next()
s.run_attack()
```
That a package provides a hooking mechanism is convenient, but the user also has to be aware of the what to pass to the hooking mechanism as well as what format to pass it in.

Note that `pytorch` also provides other hooking functionality such as:
- `register_forward_pre_hook`
- `register_full_backward_hook`