# Frequently Asked Questions

#### Why do Datasets return NumPy Arrays?
Currently, the ART toolbox only accepts NumPy arrays as inputs to attacks, defenses 
and model fitting. It is understood that this is an inefficient way to utilize data 
since it requires a conversion from FrameworkTensor -> ndarray -> FrameworkTensor.

Framework specific attacks / defenses are on the roadmap for ART and when available we 
will switch to having PyTorch or TensorFlow data generators.


#### Accessing underlying wrapped model
There are many times when creating a scenario you may want to access the underlying 
framework model that has been wrapped as an ART classifier. In the future we'll have 
a convience method to access the models through an ART api, but in the short term they 
can be accessed as follows:

KerasWrapper:
```
underlying_model = wrapped_classifier._model
```

PyTorchWrapper:
```
underlying_model = wrapped_classifier._model._model 
```
