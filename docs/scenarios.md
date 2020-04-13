# Scenarios
Armory is intended to evaluate threat-model scenarios.  

### Base Scenario Class
All scenarios inherit from the [Base Armory Scenario](armory/scenarios/base.py). The 
base class parses an armory configuration file and calls a particular scenario's 
private `_evaluate` to perform all of the computation for a given threat-models 
robustness to attack. All `_evaluate` methods return a  dictionary of recorded metrics 
which are saved into the armory `output_dir` upon  completion.
 
### Baseline Scenarios
Currently the following Scenarios are available within the armory package:
* [Image classification](armory/scenarios/image_classification.py)
* Audio classification
* Video classification
* Poisoned image classification

### Downloading Weights and Datasets for a scenario
As a convience we provide methods to download the datasets and model weights for 
baseline scenarios in armory ahead of running them.

```

``` 

### Creating a new scenario
User's may want to create their own scenario, because the baseline scenarios do 
not fit the requirements of some defense/threat-model, or because it may be easier 
to debug in code that you have access to as opposed to what is pre-installed by the 
armory package.

An example of doing this can be found in our armory-examples repo:
[Add link]
