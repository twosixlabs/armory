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
### Example Code<a name="example1"></a>
This is an example of working with a python package/framework (i.e. `pytorch`) that comes with built-in hooking mechanisms. In the code snippet below, we are relying on an existing function `register_forward_hook` to monitor the layer of interest:
```python
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
Aside the specifics of using `register_forward_hook`, consider how `Probe` and `Meter` are incorporated in this example. Recall the steps for a minimal working example (in [Measurement Overview](./metrics.md)):
1. Create `Probe` via `get_probe("test")`
2. Define `Probe` actions
3. Connect `Probe`
4. Create `Meter` with processing functions that take input from created `Probe`
5. Connect `Meter` to `Hub` via `get_hub().connect_meter(meter)`

#### Step 1
Note the input `"test"` that is passed in `get_probe("test")` - this needs to match with the first portion of a `.`-separated argument name `"test.layer4_2_relu"` that is passed to creating a `Meter` in [Step 3](#step4)

#### Step 2
The `update` method for `Probe` takes as input optional processing functions and variable names and corresponding values that are to be monitored.
- The variable name `layer4_2_relu` is how we are choosing to reference a certain value
    - this needs to match with the second portion of a `.`-separated argument name `"test.layer4_2_relu"` that is passed to creating a `Meter` in [Step 3](#step3)
- `hook_output[0][0]` is the value we are interested in, which is the output from `s.model.model.backbone.body.layer4[2].relu` after a forward pass
    - `[0][0]` was included to slice the output to show that it can be done, and because we know the shape of the output in advance
- `lambda x: x.detach().cpu().numpy()` is the processing function that converts `hook_output[0][0]` from a tensor to an array

#### Step 3
This particular step is not dealt with in-depth in [Measurement Overview](./metrics.md), but requires more explanation for this section.

#### Step 4<a name="step4"></a>
In this particular example, the `Meter` accepts 3 inputs: a meter name, a metric/function for processing, and a argument name to pass the metric/function.
- The meter name (`"layer4_2_relu"`) can be arbitrary within this context
- For the scope of this document, we only consider simple `Meter`s with the identity function as a metric i.e. `Meter` will record variables monitored by `Probe` as-is (thus `lambda x: x`)
- The argument passed to the metric/function follows a `.`-separated format (`"test.layer4_2_relu"`), which needs to be consistent with `Probe` setup:
    - `test` matches input in `get_probe("test")`
    - `layer4_2_relu` matches variable name in `layer4_2_relu=hook_output[0][0]`

#### Step 5
For the scope of this document, we don't dwell on what `armory` is doing in step 5 with `get_hub().connect_meter(meter)` other than to mention this step is necessary.

## Example 2: Attack Artifact - Available as Output
### User Story
I am using `CARLADapricotPatch` I am interested in the patch after every iteration, which is generated by `CARLADapricotPatch._augment_images_with_patch`
### Example Code
This is an example of working with a python package/framework (i.e. `art`) that does NOT come with built-in hooking mechanisms. In the code snippet below, we define wrapper functions to wrap existing instance methods to monitor the output of interest:
```python
from armory.scenarios.main import get as get_scenario
from armory.instrument import get_probe, Meter, get_hub
import types

def method_hook(obj, method_name, pre_method_hook=None, post_method_hook=None):
    """
    Hook target method and return the original method
    If a class is passed in, hooks ALL instances of class.
    If an object is passed in, only hooks the given instance.
    """
    if not isinstance(obj, object):
        raise ValueError(f"obj {obj} is not a class or object")
    method = getattr(obj, method_name)
    if not callable(method):
        raise ValueError(f"obj.{method_name} attribute {method} is not callable")
    wrapped = hook_wrapper(
        method, pre_method_hook=pre_method_hook, post_method_hook=post_method_hook
    )

    if isinstance(obj, type):
        cls = obj
        setattr(cls, method_name, wrapped)
    else:
        setattr(obj, method_name, types.MethodType(wrapped, obj))

    return method

def hook_wrapper(method, pre_method_hook = None, post_method_hook = None):
    def wrapped(*args, **kwargs):
        return_value = method(*args[1:], **kwargs)
        post_method_hook(*return_value)
        return return_value

    return wrapped

def post_method_hook(x_patch, patch_target, transformations):
    probe.update(x_patch=x_patch)  # adding batch dim

# load Scenario
s = get_scenario(
    "/armory/tmp/2022-11-03T180812.020999/carla_obj_det_adversarialpatch_undefended.json",
    num_eval_batches=1,
).load()

# create Probe with "hooked_method" namespace
probe = get_probe("hooked_method")

# register hook that will update Probe
method_hook(
    s.attack, "_augment_images_with_patch", post_method_hook=post_method_hook
)

# create Meter for Probe with "hooked_method" namespace
hook_meter = Meter(
    "hook_x_patch", lambda x: x, "hooked_method.x_patch"
)

# connect Meter to Hub
get_hub().connect_meter(hook_meter)

s.next()
s.run_attack()
```
### Packages with NO Hooks
Unlike [Example 1](#example1), we have defined new functions to meet user needs:
- `method_hook(obj, method_name, pre_method_hook=None, post_method_hook=None)`
- `hook_wrapper(method, pre_method_hook = None, post_method_hook = None)`
- `post_method_hook(x_patch, patch_target, transformations)`

The steps are the same as before, with the exception that [Step 3](#step3) is more involved than Example 1. 

### Probe and Meter Details - Step 3
The general approach for hooking a `Probe` is as follows:
1. Define the function for the `Probe` action (e.g. `post_method_hook`)
2. Wrap the method of interest (e.g. `_augment_images_with_patch`) and `post_method_hook`
    1. Define function (e.g. `hook_wrapper`) that returns another function (e.g. `wrapped`) that calls `_augment_images_with_patch` and `post_method_hook` in the desired order
    2. Assign the result of `hook_wrapper` to the original method of interest (`_augment_images_with_patch`) via `method_hook`, thus changing the behavior of the method without modifying it directly

#### Step 3-1: `post_method_hook`<a name="step3-1"></a>
The role of the defined `post_method_hook` function is the same as that of `hook_fn` defined in [Example 1](#example1) - we are specifying the variable to update being monitored by the `Probe`. In this particular example, despite its name, it is not the name that is specifying whether the `Probe` action occurs after a method, but which input this function is assigned to when calling `method_hook`, which we will show later.

Note the signature of `post_method_hook` with 3 arguments - this was based on the expected output of `_augment_images_with_patch` i.e. `return_value = method(*args[1:], **kwargs)`, and the anticipation that the output of `_augment_images_with_patch` would be passed to `post_method_hook` as input in `hook_wrapper` i.e. `post_method_hook(*return_value)`.

Of those expected outputs, we are choosing to update the value assigned to argument `x_patch` of `post_method_hook` and choosing to also refer to the updated value as `x_patch` by the `Probe`, which leads to `probe.update(x_patch=x_patch)`. The `Meter` is then able to reference this as `"hooked_method.x_patch"` later on.

For now, consider the example shown as a possible template for the user - we leave it as a template rather than add a defined function in armory such that the user can adjust as needed.

#### Step 3-2.1: `hook_wrapper`
`hook_wrapper` determines when the `Probe` action takes place with respect to a `method` call as well as how inputs should be specified for either a `pre_method_hook` or a `post_method_hook`. Its signature as defined suggests the possibility of either, but in this example, we only specify `post_method_hook`.

Within `hook_wrapper`, we define `wrapped`, which is where the actual order of calls for `method` and `post_method_hook` are specified, which in this case is `method` then `post_method_hook`. Again, despite the argument name, there is no reason the user cannot specify `post_method_hook` to be called before `method`, but we discourage such practices for clarity.

Now consider the arguments involved within `hook_wrapper`. `hook_wrapper` once called, will return a function `wrapped`, which expects arguments `*args` and `**kwargs`. Because `hook_wrapper` is meant to wrap `method`, `*args` and `**kwargs` will be passed to `method` such that it performs its original function. Recall that `method` is not just any ordinary function but a method for an instance, which means that even though the first argument of `*args` will be `self` <ins>***[why was `self` even passed to begin with???]***</ins>, the actual method call needs to exclude it, leading to `method(*args[1:], **kwargs)`.

Also note what `wrapped` returns. Since `method` needs to maintain its original functionality, the return value for `wrapped` should also match that of `method`, thus `return return_value`.

Last but not least, consider the `post_method_hook` call. For this example, the objective was to update a variable that is an output of `method` after each `method` call, and as mentioned in [Step 3-1](#step-3-1-post_method_hook), we pass `return_value` as-is but with a `*` to unpack the iterable, thus `post_method_hook(*return_value)`.

We present `hook_wrapper` as another template for the user rather than a defined armory function, because, as we have alluded to before, the arrangement of a `Probe` action and instance method is heavily dependent on the user's objective and not obviously generalizable as we will see in the next example.

#### Step 3-2.2: `method_hook`
Notice that `hook_wrapper` returns a wrapped method, but the wrapped method is not actually reassigned to the method that was meant to be wrapped. `method_hook` is defined to do just that, along with actually executing `hook_wrapper` as well for the wrapping process.

Unlike `post_method_hook` and `hook_wrapper`, which we made a point of framing as templates, we believe `method_hook` is well-established and generalized enough to be defined as an armory function, which the user can import and use as-is.