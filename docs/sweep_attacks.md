# Sweep Attacks

Armory supports running adversarial attacks which "sweep" over a range of values for specified 
attack parameters (e.g. `"eps"`). This helps automate the process of determining at what 
attack parameter values (i.e. perturbation budget) a defense is no longer robust. The attack
returns the weakest-strength adversarial example that is successful, or the original input 
if the attack fails at all values.

To enable such an attack, set `attack_config["type"]` to `"sweep"`. 
```aidl
"attack": {
    "module": "art.attacks.evasion",
    "name": "ProjectedGradientDescent",
    ...
    "type": "sweep"
}
```

Next, specify which parameter(s) to perform the search over. This is done using the 
`attack_config["sweep_params"]["kwargs"]` field for kwargs passed to attack instantiation and the
`attack_config["sweep_params"]["generate_kwargs"]` field for kwargs passed to the attack's 
`generate()` method. In the example below we sweep over the `"eps"` and `"eps_step"` 
parameters:

```aidl
"attack": {
    "module": "art.attacks.evasion",
    "name": "ProjectedGradientDescent",
    "type": "sweep",
    "sweep_params": {
        "kwargs": {
            "eps": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "eps_step": [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        }
    }
}
```

Similarly, in the following example, we sweep over kwargs passed to the attack's `generate()` method 
using the `"generate_kwargs"` field inside `attack_config["sweep_params"]`:
```aidl
"attack": {
    "module": "armory.art_experimental.attacks.pgd_patch",
    "name": "PGDPatch",
    "type": "sweep",
    "sweep_params": {
        "generate_kwargs": {
            "patch_height": [10, 20, 30, 40, 50],
            "patch_width": [10, 20, 30, 40, 50]
        }
    }
}
```

Each range of sweep parameters must be specified with a list of length `N`, where `N > 1` is 
the number of desired search points. All parameters specified inside of 
`attack_config["sweep_params"]` must correspond to lists of length `N`, and the `ith` element of
a given kwarg list will be used in conjunction with the `ith` element of all other kwarg 
lists (i.e. in the first example, when `"eps"` is `0.01`, `"eps_step"` is `0.005` and so on). The
search algorithm assumes that parameters lists are in ascending order of attack strength.

Attack parameters to be held constant should be specified in `attack_config["kwargs"]` and
`attack_config["generate_kwargs"]` as per usual. If the same kwarg (or generate_kwarg) appears in `attack_config["kwargs"]`
and `attack_config["sweep_params"]["kwargs"]`, the former will be ignored. 


### Determining Attack Success
In order to identify at what point an attack is successful, it is necessary to define how
attack success is measured. By default `armory.utils.metrics.categorical_accuracy` is used to 
determine whether the predicted label matches the ground-truth. For non-classification 
scenarios such as object detection or speech recognition, this metric doesn't apply. 

Users can specify the task-relevant metric used to measure robustness via the 
`attack_config["sweep_params"]["metric"]` field. Inside this field, specify a `"module"` and
`"name"` which point to the desired metric function. This can be any function `f` which takes
positional arguments `y` and `y_pred` as such: `f(y, y_pred)` and returns a scalar. A 
`"threshold"` must also be specified indicating at what metric value that attack is 
considered successful. For non-targeted attacks, if the value is *below* the threshold we 
consider the attack successful, while the opposite is true for targeted attacks.

### Additional Configuration Settings
Sweep attacks require access to either ground-truth or target labels `y`. If the attack 
is untargeted, set `attack_config["use_label"]` to `true`.


To ensure that metrics are saved on a per-example basis, set 
`metric_config["record_metric_per_sample"]` to `true`.
