# Scenarios
Armory is intended to evaluate threat-model scenarios. Baseline evaluation scenarios 
are described below. Additionally, we've provided some academic standard scenarios.

## Base Scenario Class
All scenarios inherit from the [Base Armory Scenario](../armory/scenarios/base.py). The 
base class parses an armory configuration file and calls a particular scenario's 
private `_evaluate` to perform all of the computation for a given threat-models 
robustness to attack. All `_evaluate` methods return a  dictionary of recorded metrics 
which are saved into the armory `output_dir` upon  completion.
 
## Baseline Scenarios
Currently the following Scenarios are available within the armory package.

### RESISC image classification

* **Description:** 
* **Threat Scenario:**
* **Metrics of Interest:**
* **Baseline Model Performance:**
* **Baseline Defense Performance:**


### Librispeech speaker audio classification

* **Description:** 
* **Threat Scenario:**
* **Metrics of Interest:**
* **Baseline Model Performance:**
* **Baseline Defense Performance:**:

### UCF101 video classification

* **Description:** 
* **Threat Scenario:**
* **Metrics of Interest:**
* **Baseline Model Performance:**
* **Baseline Defense Performance:**

### German traffic sign poisoned image classification


* **Description:** 
* **Threat Scenario:**
* **Metrics of Interest:**
* **Baseline Model Performance:**
* **Baseline Defense Performance:**

## Academic Scenarios

### Cifar10 image classification

* **Description:** This is a standard white-box attack scneario. 
* **Threat Scenario:** White-box attack
* **Metrics of Interest:** Benign accuracy, Adversarial accuracy, Adversarial purturbation
* **Baseline Model Performance:** 
* **Baseline Defense Performance:** See academic literature for the most up to date results

### MNIST image classification

* **Description:**
* **Threat Scenario:** White-box attack 
* **Metrics of Interest:** Benign accuracy, Adversarial accuracy, Adversarial purturbation
* **Baseline Model Performance:** 
* **Baseline Defense Performance:** See academic literature for the most up to date results

## Creating a new scenario
Users may want to create their own scenario, because the baseline scenarios do 
not fit the requirements of some defense/threat-model, or because it may be easier 
to debug in code that you have access to as opposed to what is pre-installed by the 
armory package.

An [example of doing this](https://github.com/twosixlabs/armory-example/blob/master/example_scenarios/audio_spectrogram_classification.py) can be found in our armory-examples repo:
