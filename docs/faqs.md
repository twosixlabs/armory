# Frequently Asked Questions

### What do the armory version strings mean?

As of Armory 0.16 we are using a newer mechanism for versioning which is
intended to help armory developers and some users better record the precise
code commit used in a build.

In the normal case, as a regular armory user,

    pip install armory-testbed
    armory --version

will yield an undecorated version number like `0.16.0`. This version also
gets recorded in armory output files, so if you are running a clean release version,
you won't see anything different.

If you modify the armory source and then rebuild it, the version will contain a
git commit id. For example, my most recent build when I modified my local source
was

    0.16.0.builda7492e4

This shows that the build was made on a 0.16.0 base with the most recent commit
being `456abc`. This is useful for developers who want to know exactly what code
produced this build. It also allows recreation of a prior version if needed by

    git checkout a7492e4
    pip install -e .

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
