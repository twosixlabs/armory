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
In this scenario, the system under evaluation is assumed to be a real-time overhead imagery scene classification
system that a human operator is either passively monitoring or not monitoring at all.
* **Dataset:**
The dataset is the [NWPU RESISC-45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).
It comprises 45 classes and 700 images for each class.  Images 1-500 of each class are in the training split,
500-600 are in the validation split, and 600-700 are in the test split.    
* **Baseline Model:**
To maximize time spent on defense research, a trained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model is an ImageNet-pretrained DenseNet-121 that is fine-tuned on RESISC-45.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to evade detection
    * Targeted - an adversary may wish to divert attention or resources to scenes that are otherwise uninteresting
  * Adversary Operating Environment:
    * Non real-time, digital evasion attack - attack is not "per-example" bur "universal," which could be created
    offline (i.e., non real-time). The goal is to mimic conditions under which physical evasion attack could be carried out.
    * Black-box, white-box, and adaptive attacks will be performed on defenses - for black-box attack, a held-back
    model or dataset will be used as surrogate.
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under quick glance are allowed - we assume in this scenario that 
    a human may at most passively monitor the classifier system. Use own judgement on the maximum perturbation 
    budget allowed while meeting the perceptability requirement.
    * Type of attacks that will be implemented during evaluation: universal perturbation (untargeted attack) and 
    universal patch (targeted attack)
      * For universal patch attack, assume the total area of the patch is at most 25% of the total image area.  The 
      location and shape of the patch will vary.
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack computational cost, defense computational cost, various distance measures of perturbation 
    (Lp-norms, Wasserstein distance)
  * Derivative metrics - see end of document 
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Model Performance:**
  * Baseline Clean Top-1 Accuracy: 93%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 6%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 23%
* **Baseline Defense Performance:**
Baseline defense is art_experimental.defences.JpegCompressionNormalized(clip_values=(0.0, 1.0), quality=50, channel_index=3, apply_fit=False,
apply_predict=True, means=[0.36386173189316956, 0.38118692953271804, 0.33867067558870334], stds=[0.20350874, 0.18531173, 0.18472934]) - see
resisc45_baseline_densenet121_adversarial.json for example usage.
Baseline defense performance is evaluated for a grey-box attack: adversarial examples generated on undefended baseline model evaluated on defended model.
  * Baseline Clean Top-1 Accuracy: 92%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 40%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 21%

### Librispeech speaker audio classification

* **Description:**
In this scenario, the system under evaluation is a speaker identification system that a human operator is either
passively monitoring or not monitoring at all.
* **Dataset:**
The dataset is the [LibriSpeech dataset](http://www.openslr.org/12).
Due to the large size of the dataset, a custom subset is created.  It comprises 40 speakers (20 male/ 20 female), 
each with 4/2/2 minutes of audio in the train/validation/test splits, respectively.
* **Baseline Model:**
To maximize time spent on defense research, two trained baseline models will be provided - one based on spectrogram (not 
mel-cepstrum or MFCC) and one based on raw audio - but performers are not required to use them, if their defense 
requires a different architecture. The spectrogram-based model is developed and trained from scratch, and the 
raw audio-based model is [SincNet](https://arxiv.org/abs/1808.00158), trained from scratch.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to evade detection
    * Targeted - an adversary may wish to impersonate someone else
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack - attack is not "per-example" bur rather "universal," which could be created
    offline (i.e., non real-time). The goal is to mimic conditions under which physical evasion attack could be carried out.
    * Assuming perfect acoustic channel
    * Black-box, white-box, and adaptive attacks will be performed on defenses - for black-box attack, spectrogram-based
    model will be the surrogate for the raw audio-based model, and vice versa.
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under passive listening are allowed - we assume in this scenario that
    a human may at most passively monitor the classifier system. Use own judgement on the maximum perturbation budget 
    allowed while meeting the perceptability requirement.
    * Type of attacks that will be implemented during evaluation: universal perturbation (untargeted and targeted attacks)
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack computational cost, defense computational cost, various distance measures of perturbation
    (Lp-norms, Wasserstein distance, signal-to-noise ratio)
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Model Performance:**
To be added
* **Baseline Defense Performance:**
To be added


### UCF101 video classification

* **Description:**
In this scenario, the system under evalution is a video action recognition system that a human operator is either
passively monitoring or not monitoring at all.
* **Dataset:**
The dataset is the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php).
It comprises 101 actions and 13,320 total videos. For the training/testing split,
we use the official Split 01.
* **Baseline Model:**
To maximize time spent on defense research, a trained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model uses the [MARS architecture](http://openaccess.thecvf.com/content_CVPR_2019/papers/Crasto_MARS_Motion-Augmented_RGB_Stream_for_Action_Recognition_CVPR_2019_paper.pdf),
which is a single-stream (RGB) 3D convolution architecture that simultaneously mimics the optical flow stream. 
The provided model is pre-trained on the Kinetics dataset and fine-tuned on UCF101.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to evade detection
    * Targeted - an adversary may wish to divert attention or resources to videos that are otherwise uninteresting
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack - we assume an adversary is the author of the video, so they could create an evasion attack offline
    before distributing the video.  Typically, a non real-time attack is "universal," but in this case, it is "per-example."
    * Black-box, white-box, and adaptive attacks will be performed on defenses - for black-box attack, a held-back
    model or dataset will be used as surrogate.
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under quick glance are allowed, as are attacks that create perceptible
    but non-suspicious patches - we assume in this scenario that a human may at most passively monitor the classifier system.
    Use own judgement on the maximum perturbation budget allowed while meeting the perceptability requirement.
    * Type of attacks that will be implemented during evaluation: perturbation (untargeted attack) and
    patch (targeted attack)
      * For patch attack, assume the total area of the patch is at most 25% of the total image area.  The 
      location and shape of the patch will vary.
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack computational cost, defense computational cost, various distance measures of perturbation
    (Lp-norms, Wasserstein distance)
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Model Performance:**
  * Baseline Clean Top-1 Accuracy: 93%
  * Baseline Clean Top-5 Accuracy: 99%
  * Baseline Attacked (Perturbation) Top-1 Accuracy: 4%
  * Baseline Attacked (Perturbation) Top-5 Accuracy: 35%
  * Baseline Attacked (Patch) Top-1 Accuracy: 24%
  * Baseline Attacked (Patch) Top-5 Accuracy: 97%
* **Baseline Defense Performance:**
Baseline defense is art_experimental.defences.JpegCompression5D(clip_values=(0.0, 255.0), quality=50, channel_index=3, apply_fit=False,
apply_predict=True, means=[114.7748, 107.7354, 99.475], transpose=[1, 2, 3, 0]) - see ucf101_baseline_adversarial.json for example usage.
Baseline defense performance is evaluated for a grey-box attack: adversarial examples generated on undefended baseline model evaluated on defended model.
  * Baseline Clean Top-1 Accuracy: 88%
  * Baseline Clean Top-5 Accuracy: 98%
  * Baseline Attacked (Perturbation) Top-1 Accuracy: 65%
  * Baseline Attacked (Perturbation) Top-5 Accuracy: 96%
  * Baseline Attacked (Patch) Top-1 Accuracy: 86%
  * Baseline Attacked (Patch) Top-5 Accuracy: 97%

### German traffic sign poisoned image classification

* **Description:**
In this scenario, the system under evalution is a traffic sign recognition system that requires continuous
training, and the training data is procured through less trustworthy external sources (e.g., third-party, Internet, etc.)
and may contain backdoor triggers, where some images and labels are intentionally altered to mislead the system into 
making specific test-time decisions.
* **Dataset:**
The dataset is the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
It comprises 43 classes and more than 50,000 total images. The official Final_Training and Final_Test data are used
for the train/test split.
* **Baseline Model:**
To maximize time spent on defense research, an untrained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model uses the [MicronNet architecture](https://arxiv.org/abs/1804.00497). Also provided will be
poison data (1/5/10% of the training size) that should be mixed with the training data.
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack - the goal is to mimic conditions under which physical evasion attack 
    could be carried out.
    * Black-box, white-box, and adaptive attacks will be performed on defenses - for black-box attack, a held-back
    model or dataset will be used as surrogate.
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under quick glance are allowed, as are attacks that create perceptible
    but non-suspicious triggers - we assume in this scenario that a human may at most passively monitor the classifier system.
  * The type of attack will be a trigger-based one where the inputs and labels are modified.
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), backdoor success rate, attack computational cost, defense computational cost
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Model Performance:**
To be added
* **Baseline Defense Performance:**
To be added

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

## Derivative metrics
![alt text](https://user-images.githubusercontent.com/18154355/80718651-691fb780-8ac8-11ea-8dc6-94d35164d494.png "Derivative Metrics")
