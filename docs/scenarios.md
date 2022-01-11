# Scenarios
Armory is intended to evaluate threat-model scenarios. Baseline evaluation scenarios 
are described below. Additionally, we've provided some academic standard scenarios.

## Base Scenario Class
All scenarios inherit from the [Base Armory Scenario](https://github.com/twosixlabs/armory/blob/master/armory/scenarios/base.py). The 
base class parses an armory configuration file and calls a particular scenario's 
private `_evaluate` to perform all of the computation for a given threat-models 
robustness to attack. All `_evaluate` methods return a  dictionary of recorded metrics 
which are saved into the armory `output_dir` upon  completion.
 
## Baseline Scenarios
Currently the following Scenarios are available within the armory package.

### RESISC image classification (Updated June 2020)

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
    budget allowed while meeting the perceptibility requirement.
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
* **Baseline Model Performance: (results obtained using Armory < v0.10)**
  * Baseline Clean Top-1 Accuracy: 93%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 6%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 23%
* **Baseline Defense Performance: (results obtained using Armory < v0.10)**
Baseline defense is art_experimental.defences.JpegCompressionNormalized(clip_values=(0.0, 1.0), quality=50, channel_index=3, apply_fit=False,
apply_predict=True, means=[0.36386173189316956, 0.38118692953271804, 0.33867067558870334], stds=[0.20350874, 0.18531173, 0.18472934]) - see
resisc45_baseline_densenet121_adversarial.json for example usage.
Baseline defense performance is evaluated for a grey-box attack: adversarial examples generated on undefended baseline model evaluated on defended model.
  * Baseline Clean Top-1 Accuracy: 92%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 40%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 21%

### Librispeech speaker audio classification (Updated June 2020)

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
    allowed while meeting the perceptibility requirement.
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


### UCF101 video classification (Updated July 2021)

* **Description:**
In this scenario, the system under evaluation is a video action recognition system that a human operator is either
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
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack - we assume an adversary is the author of the video, so they could create an evasion attack offline
    before distributing the video.  Typically, a non real-time attack is "universal," but in this case, it is "per-example."
    * Adaptive attacks will be performed on defenses
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under quick glance are allowed, as are attacks that create perceptible
    but non-suspicious patches - we assume in this scenario that a human may at most passively monitor the classifier system.
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack budget
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Frame Saliency](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/frame_saliency.py)
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Flicker Attack](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/over_the_air_flickering/over_the_air_flickering_pytorch.py)
  * [Custom Frame Border attack](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/attacks/video_frame_border.py)
* **Baseline Defense**: [Video Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/video_compression.py)
* **Baseline Model Performance: (results obtained using Armory v0.13.3)**

|                Attack               |              Budget              | Benign Top1/Top5 Accuracy (Undefended) | Adversarial Top1/Top5 Accuracy (Undefended) | Benign Top1/Top5 Accuracy (Defended) | Adversarial Top1/Top5 Accuracy (Defended) | Test Size |
|:-----------------------------------:|:--------------------------------:|:--------------------------------------:|:-------------------------------------------:|:------------------------------------:|:-----------------------------------------:|:---------:|
| Flicker (low perceptibility)        | beta_0=4.0 beta_1=0.1 beta_2=0.9 | 0.92/1.00                              | 0.51/1.00                                   | 0.92/1.00                            | 0.44/1.00                                 | 100       |
| Flicker (medium perceptibility)     | beta_0=2.0 beta_1=0.1 beta_2=0.9 | 0.92/1.00                              | 0.39/1.00                                   | 0.92/1.00                            | 0.40/0.97                                 | 100       |
| Flicker (high perceptibility)       | beta_0=1.0 beta_1=0.1 beta_2=0.9 | 0.92/1.00                              | 0.37/1.00                                   | 0.92/1.00                            | 0.38/0.98                                 | 100       |
| Frame Border                        | patch ratio=0.10                 | 0.92/1.00                              | 0.00/0.25                                   | 0.93/1.00                            | 0.03/0.36                                 | 100       |
| Frame Border                        | patch ratio=0.15                 | 0.92/1.00                              | 0.00/0.19                                   | 0.93/1.00                            | 0.01/0.29                                 | 100       |
| Frame Border                        | patch ratio=0.20                 | 0.92/1.00                              | 0.00/0.19                                   | 0.93/1.00                            | 0.00/0.25                                 | 100       |
| Masked PGD                          | patch ratio=0.10                 | 0.92/1.00                              | 0.02/0.61                                   | 0.93/1.00                            | 0.01/0.66                                 | 100       |
| Masked PGD                          | patch ratio=0.15                 | 0.92/1.00                              | 0.00/0.42                                   | 0.93/1.00                            | 0.00/0.36                                 | 100       |
| Masked PGD                          | patch_ratio=0.20                 | 0.92/1.00                              | 0.00/0.28                                   | 0.93/1.00                            | 0.00/0.31                                 | 100       |
| Frame Saliency (iterative_saliency) | eps=0.004                        | 0.92/1.00                              | 0.00/0.96                                   | 0.92/1.00                            | 0.81/1.00                                 | 100       |
| Frame Saliency (iterative_saliency) | eps=0.008                        | 0.92/1.00                              | 0.00/0.96                                   | 0.92/1.00                            | 0.47/1.00                                 | 100       |
| Frame Saliency (iterative_saliency) | eps=0.015                        | 0.92/1.00                              | 0.00/0.96                                   | 0.92/1.00                            | 0.23/0.99                                 | 100       |
| Frame Saliency (one_shot)           | eps=0.004                        | 0.92/1.00                              | 0.00/0.26                                   | 0.93/1.00                            | 0.79/0.97                                 | 100       |
| Frame Saliency (one_shot)           | eps=0.008                        | 0.92/1.00                              | 0.00/0.22                                   | 0.93/1.00                            | 0.46/0.89                                 | 100       |
| Frame Saliency (one_shot)           | eps=0.015                        | 0.92/1.00                              | 0.00/0.20                                   | 0.93/1.00                            | 0.21/0.74                                 | 100       |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)

### German traffic sign poisoned image classification (Updated December 2020)

* **Description:**
In this scenario, the system under evaluation is a traffic sign recognition system that requires continuous
training, and the training data is procured through less trustworthy external sources (e.g., third-party, Internet, etc.)
and may contain backdoor triggers, where some images and labels are intentionally altered to mislead the system into 
making specific test-time decisions.
* **Dataset:**
The dataset is the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
It comprises 43 classes and more than 50,000 total images. The official Final_Training and Final_Test data are used
for the train/test split. The dataset is available in canonical format, but the official scenario applies preprocessing
consisting of contrast equalization, cropping to a square shape, and resizing.
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
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), backdoor success rate, attack computational cost, defense computational cost
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Dirty-label Backdoor Attack](https://arxiv.org/abs/1708.06733): 1 to 10% of a *source class* in the 
  training data have trigger added and are intentionally mislabeled with *target label*; during test time,
   the same trigger is added to an input of *source class* to cause targeted misclassification.
  * [Clean-label Backdoor Attack](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf): 1 to 10% of the 
  *target class* in training data are imperceptibly perturbed (so they are still correctly labeled) and have 
  trigger added; during test time, same trigger is added to an input of a *source class* to cause 
  targeted misclassification
    * Perturbation constraints: Linf (eps <= 16/255), L2 (eps <= 8/255 * sqrt(N)), N=# of pixels in a 
    single input
* **Baseline Defense**: [Activation Clustering](https://arxiv.org/abs/1811.03728) and/or
  [Spectral Signature](https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf)
* **Baseline Model Performance:**
To be added
* **Baseline Defense Performance:**
To be added

### Librispeech automatic speech recognition (Updated July 2021)

* **Description:**
In this scenario, the system under evaluation is an automatic speech recognition system that a human operator is either
passively monitoring or not monitoring at all.
* **Dataset:**
The dataset is the [LibriSpeech dataset](http://www.openslr.org/12) and comprises train_clean100, 
train_clean360 and test_clean.
* **Baseline Model:**
To maximize time spent on defense research, a trained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model uses the [DeepSpeech 2](https://arxiv.org/pdf/1512.02595v1.pdf) architecture with
pretrained weights from either the AN4, LibriSpeech, or TEDLIUM datasets.  Custom weights
may also be loaded by the model.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish for speech to be transcribed incorrectly
    * Targeted - an adversary may wish for specific strings to be predicted
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack. Each attack will be "universal" with respect to 
    channel conditions (under a single perfect channel, the attack will be "per-example.")
    * Under some threat models, the channel model consists only a single perfect acoustic channel, and under others, it may consist of one additional multipath channel.
    * Adaptive attacks will be performed on defenses.
  * Adversary Capabilities and Resources
    * To place an evaluation bound on the perceptibility of perturbations, the SNR is restricted to >20 dB.
* **Metrics of Interest:**
  * Primary metrics:
    * Word error rate, SNR
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Imperceptible ASR attack](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr.py)
  * [PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Kenansville attack](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/attacks/kenansville_dft.py)
* **Baseline Defense**: [MP3 Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/mp3_compression.py)
* **Baseline Model Performance: (results obtained using Armory v0.13.3)**

| Attack                   | Targeted | Budget         | Benign WER (Undefended) | Adversarial WER (Undefended) | Benign WER (Defended) | Adversarial WER (Defended) | Test Size |
|--------------------------|----------|----------------|-------------------------|------------------------------|-----------------------|----------------------------|-----------|
| Imperceptible ASR        | yes      | max_iter_1=100 | 0.10                    | 0.63                         | 0.13                  | N/A*                       | 320       |
| Imperceptible ASR        | yes      | max_iter_1=200 | 0.10                    | 0.20                         | 0.13                  | N/A                        | 320       |
| Imperceptible ASR        | yes      | max_iter_1=400 | 0.10                    | 0.11                         | 0.13                  | N/A                        | 320       |
| Kenansville              | no       | snr=20dB       | 0.10                    | 0.27                         | 0.13                  | 0.36                       | 1000      |
| Kenansville              | no       | snr=30dB       | 0.10                    | 0.11                         | 0.13                  | 0.17                       | 1000      |
| Kenansville              | no       | snr=40dB       | 0.10                    | 0.10                         | 0.13                  | 0.13                       | 1000      |
| PGD (single channel)     | no       | snr=20dB       | 0.10                    | 0.46                         | 0.13                  | 0.53                       | 100       |
| PGD (single channel)     | no       | snr=30dB       | 0.10                    | 0.46                         | 0.13                  | 0.50                       | 100       |
| PGD (single channel)     | no       | snr=40dB       | 0.10                    | 0.33                         | 0.13                  | 0.36                       | 100       |
| PGD (single channel)*    | yes      | snr=20dB       | 0.11                    | 1.03                         | 0.15                  | 1.01                       | 100       |
| PGD (single channel)*    | yes      | snr=30dB       | 0.11                    | 1.02                         | 0.15                  | 0.99                       | 100       |
| PGD (single channel)*    | yes      | snr=40dB       | 0.11                    | 0.88                         | 0.15                  | 0.84                       | 100       |
| PGD (multiple channels)  | no       | snr=20dB       | 0.13                    | 0.96                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)  | no       | snr=30dB       | 0.13                    | 0.59                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)  | no       | snr=40dB       | 0.13                    | 0.38                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=20dB       | 0.13                    | 0.99                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=30dB       | 0.13                    | 0.92                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=40dB       | 0.13                    | 0.75                         | N/A                   | N/A                        | 100       |

* \*Target attack, where a random target phrase of similar length as the ground truth, was applied but WER wrt the ground truth was calculated

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)
* Missing defended baseline is due to current incompatibility of the attack and defense.
  

### so2sat multimodal image classification (Updated July 2021)

* **Description:**
In this scenario, the system under evaluation is an image classifier which determines local climate zone from a combination of co-registered synthetic aperture radar (SAR) and multispectral electro-optical (EO) images.
* **Dataset:**
The dataset is the [so2sat dataset](https://mediatum.ub.tum.de/1454690). It comprises 352k/24k images in
train/validation datasets and 17 classes of local climate zones.
* **Baseline Model:**
To maximize time spent on defense research, a trained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model uses a custom CNN architecture with a single input that stacks SAR (first four channels only,
representing the real and imaginary components of the reflected electromagnetic waves) 
and EO (all ten channels) data. Immediately after the input layer, the data is split into SAR and EO data 
streams and fed into their respective feature extraction networks. In the final layer, the two
networks are fused to produce a single prediction output.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary wish to evade correct classification
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack
    * Adversary perturbs a single modality (SAR or EO)
    * Adaptive attacks will be performed on defenses.
  * Adversary Capabilities and Resources
    * Patch ratio < 15% of the image area
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), Patch size
  * Derivative metrics - see end of document 
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
* **Baseline Defense**: [JPEG Compression for Multi-Channel](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/defences/jpeg_compression_multichannel_image.py)
* **Baseline Model Performance: (results obtained using Armory v0.13.3)**

| Attacked Modality | Patch Ratio | Benign Accuracy (Undefended) | Adversarial Accuracy (Undefended) | Benign Accuracy (Defended) | Adversarial Accuracy (Defended) | Test Size |
|-------------------|-------------|------------------------------|-----------------------------------|----------------------------|---------------------------------|-----------|
| EO                | 0.05        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |
| EO                | 0.10        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |
| EO                | 0.15        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |
| SAR               | 0.05        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |
| SAR               | 0.10        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |
| SAR               | 0.15        | 0.583                        | 0.00                              | 0.556                      | 0.00                            | 1000      |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)

### xView object detection (Updated July 2021)

* **Description:**
In this scenario, the system under evaluation is an object detector which localizes and identifies various classes from satellite imagery.
* **Dataset:**
The dataset is the [xView dataset](https://arxiv.org/pdf/1802.07856). It comprises 59k/19k train and test 
images (each with dimensions 300x300, 400x400 or 500x500) and 62 classes
* **Baseline Model:**
To maximize time spent on defense research, a trained baseline model will be provided, but
performers are not required to use it, if their defense requires a different architecture.
The model uses the [Faster-RCNN ResNet-50 FPN](https://arxiv.org/pdf/1506.01497.pdf) architecture pre-trained
on MSCOCO objects and fine-tuned on xView.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary wishes to disable object detection
  * Adversary Operating Environment:
    * Non-real time, digital and physical-like evasion attacks 
    and translation.
    * Adaptive attacks will be performed on defenses.
* Adversary Capabilities and Resources
    * Patch size <100x100 pixels
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of ground truth classes, Patch Size
  * Derivative metrics - see end of document 
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Robust DPatch](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch_robust.py)
* **Baseline Defense**: [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Model Performance: (results obtained using Armory v0.13.3)**

|     Attack    | Patch Size | Benign mAP (Undefended) | Adversarial mAP (Undefended) | Benign mAP (Defended) | Adversarial mAP (Defended) | Test Size |
|:-------------:|:----------:|:-----------------------:|:----------------------------:|:---------------------:|:--------------------------:|:---------:|
| Masked PGD    | 50x50      | 0.284                   | 0.142                        | 0.232                 | 0.139                      | 100       |
| Masked PGD    | 75x75      | 0.284                   | 0.071                        | 0.232                 | 0.094                      | 100       |
| Masked PGD    | 100x100    | 0.284                   | 0.076                        | 0.232                 | 0.092                      | 100       |
| Robust DPatch | 50x50      | 0.284                   | 0.193                        | 0.232                 | 0.184                      | 100       |
| Robust DPatch | 75x75      | 0.284                   | 0.184                        | 0.232                 | 0.146                      | 100       |
| Robust DPatch | 100x100    | 0.284                   | 0.173                        | 0.232                 | 0.165                      | 100       |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)

### DAPRICOT object detection (Updated July 2021)

* **Description:**
In this scenario, the system under evaluation is an object detector trained to identify the classes in the [Microsoft COCO dataset](https://arxiv.org/pdf/1405.0312.pdf).
* **Dataset:**
The dataset is the [Dynamic APRICOT (DAPRICOT) dataset 1](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/data/adversarial/dapricot_dev.py) and [dataset 2](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/data/adversarial/dapricot_test.py). It is similar to the APRICOT dataset (see below), but instead of pre-generated physical patches taken in the natural environment, the DAPRICOT dataset contains greenscreens and natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed. This dataset contains 15 scenes, where each scene contains 3 different greenscreen shapes, taken at 3 different distances, 3 different heights and using 3 different camera angles, for a total of over 1000 images.
* **Baseline Model:**
The model uses the pretrained [Faster-RCNN with ResNet-50](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) model.
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted attack - objective is to force an object detector to localize and classify the patch as an MSCOCO object.
  * Adversary Operating Environment:
    * Non-real time, digital and physical-like patch attacks
    * Adaptive attacks will be performed on defenses.
* Adversary Capabilities and Resources
    * Patch size of different shapes as dictated by the greenscreen sizes in the images
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of patches, Average target success
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Robust DPatch](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch_robust.py)
* **Baseline Defense**: [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Model Performance: (results obtained using Armory v0.13.3 and [dev test data](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/data/adversarial/dapricot_test.py))**

| Attack        | Patch Size | Target Success (Undefended) | Target mAP (Undefended) | Target Success (Defended) | Target mAP (Defended) | Test Size |
|---------------|------------|-----------------------------|-------------------------|---------------------------|-----------------------|-----------|
| Masked PGD    | all        | 0.99                        | 0.91                    | 0.99                      | 0.91                  | 100       |
| Masked PGD    | small      | 0.97                        | 0.91                    | 0.97                      | 0.91                  | 100       |
| Masked PGD    | medium     | 1.00                        | 1.00                    | 1.00                      | 0.91                  | 100       |
| Masked PGD    | large      | 1.00                        | 1.00                    | 1.00                      | 0.91                  | 100       |
| Robust DPatch | all        | 0.56                        | 0.64                    | 0.61                      | 0.64                  | 100       |
| Robust DPatch | small      | 0.51                        | 0.64                    | 0.60                      | 0.64                  | 100       |
| Robust DPatch | medium     | 0.61                        | 0.64                    | 0.65                      | 0.73                  | 100       |
| Robust DPatch | large      | 0.55                        | 0.64                    | 0.63                      | 0.73                  | 100       |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)
  
### APRICOT object detection (Updated December 2020)

* **Description:**
In this scenario, the system under evaluation is an object detector trained to identify the classes in the [Microsoft COCO dataset](https://arxiv.org/pdf/1405.0312.pdf).
* **Dataset:**
The dataset is the [APRICOT dataset](https://arxiv.org/pdf/1912.08166.pdf), which includes over 1000 natural images with physically-printed adversarial patches,
covering three object detection architectures (Faster-RCNN with ResNet-50, SSD with MobileNet, and RetinaNet),
two shapes (circle and rectangular), and ten MS-COCO classes as targets.
* **Baseline Model:**
The model uses the pretrained [Faster-RCNN with ResNet-50, SSD with MobileNet, and RetinaNet](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) models.
Note: currently, only Tensorflow Faster-RCNN with ResNet-50 is implemented, with the other two architectures 
to be implemented in the near future. In order to perform as close to a white-box evaluation as possible,
it is strongly recommended, but not required, that performers adopt one of the above architectures for defense
research - the pretrained weights may not be robust, so performers can change the weights.
* **Threat Scenario:**
  * Adversary Operating Environment:
    * This is a dataset of precomputed adversarial images on which trained models will be evaluated.
    * Each patch is a targeted attack, whose objective is to force an object detector to localize and classify
    the patch as an MSCOCO object.
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of patches, Average precision of MSCOCO objects
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * The patches were generated using variants of [ShapeShifter](https://arxiv.org/abs/1804.05810)
* **Baseline Defense**: JPEG Compression
* **Baseline Model Performance: (results obtained using Armory v0.13)**
  * Baseline MSCOCO Objects mAP: 8.76% (all test examples)
  * Baseline Targeted Patch mAP: 5.70% (all test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.13)**\
Baseline defense is art_experimental.defences.jpeg_compression_normalized(clip_values=(0.0, 1.0), quality=10,
channel_index=3, apply_fit=False, apply_predict=True).\
Baseline defense performance is evaluated for a transfer attack.
  * Baseline MSCOCO Objects mAP: 7.83% (all test examples)
  * Baseline Targeted Patch mAP: 4.59% (all test examples)

### CARLA object detection (Updated January 2022)
* **Description:**
In this scenario, the system under evaluation is an object detector trained to identify vehicles, pedestrians, and traffic lights.
* **Dataset:**
The development dataset is the [CARLA Object Detection dataset](https://carla.org), which includes RGB and depth channels for 165 synthetic images of driving scenes, each of
which contains a green-screen intended for adversarial patch insertion. The dataset contains natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed.
* **Baseline Model:**
  * Single-modality: 
    * Pretrained [Faster-RCNN with ResNet-50](../armory/baseline_models/pytorch/carla_single_modality_object_detection_frcnn.py) model.
  * Multimodal:
    * Pretrained multimodal [Faster-RCNN with ResNet-50](../armory/baseline_models/pytorch/carla_multimodality_object_detection_frcnn.py) model.
* **Threat Scenario:**
  * Adversary objectives:
    * To degrade the performance of an object detector through the insertion of adversarial patches.
  * Adversary Operating Environment:
    * Non-real time, digital and physical-like patch attacks
    * Adaptive attacks will be performed on defenses.
* Adversary Capabilities and Resources
    * Patch size of different size/shape as dictated by the green-screen in each image. In the multimodal case, only RGB channels are to be perturbed.
* **Metrics of Interest:**
  * Primary metrics:
    * mAP
    * Disappearance rate
    * Hallucinations per image
    * Misclassification rate
    * True positive rate
* **Baseline Attacks:**
  * [Custom Robust DPatch with Input-Dependent Transformation and Color-Correction](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/carla_obj_det_patch.py)
* **Baseline Defense**: [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Model Performance: (results obtained using Armory v0.14.2 and [test data](https://github.com/twosixlabs/armory/blob/master/armory/data/adversarial/carla_obj_det_test.py))**

Single Modality (RGB) Object Detection
| Patch Size | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|:----------:|:-----------:|:---------------------------:|:--------------------------------:|:-------------------------------:|:---------------------------:|:----------------:|:--------------------------------:|:------------------------------------:|:-----------------------------------:|:-------------------------------:|:---------:|
|    Small   |  0.43/0.40  |          0.37/0.45          |              0.8/1.0             |            0.06/0.08            |          0.57/0.47          |     0.19/0.21    |             0.40/0.48            |                7.6/7.6               |              0.06/0.08              |            0.54/0.45            |     10    |
|   Medium   |  0.48/0.37  |          0.39/0.50          |              1.2/1.3             |            0.01/0.01            |          0.60/0.49          |     0.24/0.16    |             0.39/0.51            |               12.4/6.8               |              0.01/0.01              |            0.61/0.48            |     10    |
|    Large   |  0.38/0.31  |          0.36/0.43          |              1.0/1.0             |            0.05/0.02            |          0.59/0.55          |     0.17/0.14    |             0.37/0.34            |               10.3/10.2              |              0.05/0.02              |            0.57/0.55            |     10    |

Multimodality (RGB+depth) Object Detection
| Attacked  Modality | Patch Size | Benign  mAP | Benign  Disappearance  Rate | Benign  Hallucination  per Image | Benign  Misclassification  Rate | Benign  True Positive  Rate | Adversarial  mAP | Adversarial  Disappearance  Rate | Adversarial Hallucination  per Image | Adversarial Misclassification  Rate | Adversarial True Positive  Rate | Test Size |
|--------------------|:----------:|:-----------:|:---------------------------:|:--------------------------------:|:-------------------------------:|:---------------------------:|:----------------:|:--------------------------------:|:------------------------------------:|:-----------------------------------:|:-------------------------------:|:---------:|
| RGB                |    Small   |  0.51/0.48  |          0.35/0.35          |              0.0/0.3             |            0.05/0.05            |          0.61/0.61          |     0.48/0.48    |             0.36/0.35            |                1.2/3.1               |              0.08/0.05              |            0.56/0.61            |     10    |
| RGB                |   Medium   |  0.58/0.59  |          0.34/0.34          |              0.5/0.8             |             0.0/0.2             |          0.66/0.64          |     0.56/0.58    |             0.34/0.34            |                1.7/1.4               |               0.0/0.02              |            0.66/0.64            |     10    |
| RGB                |    Large   |  0.46/0.46  |          0.32/0.34          |              0.5/0.8             |            0.03/0.02            |          0.64/0.64          |     0.42/0.46    |             0.32/0.34            |                1.7/0.9               |              0.03/0.02              |            0.64/0.64            |     10    |

\*a/b in the tables refer to undefended/defended performance results, respectively.

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/r0.14.2/scenario_configs)


### CARLA video tracking (Updated January 2022)
* **Description:**
In this scenario, the system under evaluation is an object tracker trained to localize pedestrians.
* **Dataset:**
The development dataset is the [CARLA Video Tracking dataset](https://carla.org), which includes 20 videos, each of
which contains a green-screen in all frames intended for adversarial patch insertion. The dataset contains natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed.
* **Baseline Model:**
  * Pretrained [GoTurn](../armory/baseline_models/pytorch/carla_goturn.py) model.
* **Threat Scenario:**
  * Adversary objectives:
    * To degrade the performance of the tracker through the insertion of adversarial patches.
  * Adversary Operating Environment:
    * Non-real time, digital and physical-like patch attacks
    * Adaptive attacks will be performed on defenses.
* Adversary Capabilities and Resources
    * Patch size of different size/shape as dictated by the green-screen in the frames. The adversary is expected to apply a constant patch across all frames in the video.
* **Metrics of Interest:**
  * Primary metrics:
    * mean IOU
* **Baseline Attacks:**
  * [Custom Adversarial Texture with Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/r0.14.2/armory/art_experimental/attacks/carla_adversarial_texture.py)
* **Baseline Defense**: [Video Compression](https://github.com/twosixlabs/armory/blob/r0.14.2/armory/art_experimental/defences/video_compression_normalized.py)
* **Baseline Model Performance: (results obtained using Armory v0.14.2 and [test data](https://github.com/twosixlabs/armory/blob/r0.14.2/armory/data/adversarial/carla_video_tracking_test.py))**

| Benign Mean IoU (Undefended) | Benign Mean IoU (Defended) | Adversarial Mean IoU (Undefended) | Adversarial Mean IoU (Defended) | Test Size |
|:----------------------------:|:--------------------------:|:---------------------------------:|:-------------------------------:|:---------:|
|             0.59             |            0.50            |                0.18               |               0.18              |     20    |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/r0.14.2/scenario_configs)

## Academic Scenarios

### Cifar10 image classification

* **Description:** This is a standard white-box attack scenario.
* **Threat Scenario:** White-box attack
* **Metrics of Interest:** Benign accuracy, Adversarial accuracy, Adversarial perturbation
* **Baseline Model Performance:** 
* **Baseline Defense Performance:** See academic literature for the most up to date results


### MNIST image classification

* **Description:**
* **Threat Scenario:** White-box attack 
* **Metrics of Interest:** Benign accuracy, Adversarial accuracy, Adversarial perturbation
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

## Exporting Samples
Scenarios can be configured to export benign and adversarial image, video, and audio samples.  This feature is enabled by setting the `export_samples` field under `scenario` in the configuration file to a non-zero integer.  The specified number of samples will be saved in the output directory for this evaluation, along with a pickle file which stores the ground truth and model output for each sample.  For video files, samples are saved both in a compressed video format and frame-by-frame.