# Frequently Asked Questions

#### Why do Datasets return NumPy Arrays?
Currently, the ART toolbox only accepts NumPy arrays as inputs to attacks, defenses 
and model fitting. It is understood that this is an inefficient way to utilize data 
since it requires a conversion from FrameworkTensor -> ndarray -> FrameworkTensor.

Framework specific attacks / defenses are on the roadmap for ART and when available we 
will switch to having PyTorch or TensorFlow data generators.


#### How/where do I perform dataset preprocessing when running a scenario?
As of Armory 0.12, dataset preprocessing should be performed inside the model. The functions 
to retrieve datasets in [armory/data/datasets.py](../armory/data/datasets.py) do each accept 
a `preprocessing_fn` kwarg that can used when loading datasets outside the context of a scenario. 
However, this kwarg is by default set to the canonical preprocessing function for each dataset and 
is not configurable when running Armory scenarios.


#### Why are datasets loaded with non-configurable "canonical" preprocessing during Armory scenarios?
Standardizing the dataset format simplifies the measuring of perturbation distance and
allows for easier comparison across different defenses and attacks.


#### Accessing underlying wrapped model
There are many times when creating a scenario you may want to access the underlying 
framework model that has been wrapped as an ART classifier. In the future we'll have 
a convenience method to access the models through an ART api, but in the short term they
can be accessed as follows:

KerasWrapper:
```
underlying_model = wrapped_classifier._model
```

PyTorchWrapper:
```
underlying_model = wrapped_classifier._model._model 
```
