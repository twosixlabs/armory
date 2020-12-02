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


### UCF101 video classification

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
    * Targeted - an adversary may wish to divert attention or resources to videos that are otherwise uninteresting
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack - we assume an adversary is the author of the video, so they could create an evasion attack offline
    before distributing the video.  Typically, a non real-time attack is "universal," but in this case, it is "per-example."
    * Black-box, white-box, and adaptive attacks will be performed on defenses - for black-box attack, a held-back
    model or dataset will be used as surrogate.
  * Adversary Capabilities and Resources
    * Attacks that are non-overtly perceptible under quick glance are allowed, as are attacks that create perceptible
    but non-suspicious patches - we assume in this scenario that a human may at most passively monitor the classifier system.
    Use own judgement on the maximum perturbation budget allowed while meeting the perceptibility requirement.
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
* **Baseline Attacks:**
  * PGD (Linf (eps <= 16/255), L2 (eps <= 8/255 * sqrt(N)), N=# of pixels in a single input)
  * Adversarial Patch (size <20% of video area)
  * [Frame Saliency](https://arxiv.org/abs/1811.11875) (Linf (eps <= 4/255))
* **Baseline Defense**: Video Compression
* **Baseline Model Performance: (Perturbation and Patch results obtained using Armory < v0.10; 
  Frame Saliency results obtained using Armory v0.12.2)**
  * Baseline Clean Top-1 Accuracy: 93% (all test examples)
  * Baseline Clean Top-5 Accuracy: 99% (all test examples)
  * Baseline Attacked (Perturbation, Linf eps=10/255) Top-1 Accuracy: 4% (all test examples)
  * Baseline Attacked (Perturbation, Linf eps=10/255) Top-5 Accuracy: 35% (all test examples)
  * Baseline Attacked (Patch, area=10%) Top-1 Accuracy: 24% (all test examples)
  * Baseline Attacked (Patch, area=10%) Top-5 Accuracy: 97% (all test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.015) Top-1 Accuracy: 0% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.015) Top-5 Accuracy: 95% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.008) Top-1 Accuracy: 0.1% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.008) Top-5 Accuracy: 95% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.004) Top-1 Accuracy: 0.3% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.004) Top-5 Accuracy: 95% (100 test examples)
* **Baseline Defense Performance: (Perturbation and Patch results obtained using Armory < v0.10;
  Frame Saliency results obtained using Armory v0.12.2)**
Baseline defense is `art_experimental.defences.video_compression_normalized(apply_fit=false, apply_predict=true,
channels_first=false, constant_rate_factor=28, video_format="avi")`
Perturbation and Patch baseline defense performance is evaluated for a grey-box attack: 
adversarial examples generated on undefended baseline model evaluated on defended model.
Frame Saliency baseline defense performance is evaluated for a white-box attack.
  * Baseline Clean Top-1 Accuracy: 88% (all test examples)\*
  * Baseline Clean Top-5 Accuracy: 98% (all test examples)\*
  * Baseline Attacked (Perturbation, Linf eps=10/255) Top-1 Accuracy: 65% (all test examples)\*
  * Baseline Attacked (Perturbation, Linf eps=10/255) Top-5 Accuracy: 96% (all test examples)\*
  * Baseline Attacked (Patch, area=10%) Top-1 Accuracy: 86% (all test examples)\*
  * Baseline Attacked (Patch, area=10%) Top-5 Accuracy: 97% (all test examples)\*
  * Baseline Attacked (Frame Saliency, Linf eps=0.015) Top-1 Accuracy: 38% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.015) Top-5 Accuracy: 99% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.008) Top-1 Accuracy: 67% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.008) Top-5 Accuracy: 100% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.004) Top-1 Accuracy: 82% (100 test examples)
  * Baseline Attacked (Frame Saliency, Linf eps=0.004) Top-5 Accuracy: 100% (100 test examples)\
\* Defended results were obtained prior to the implementation of video compression and 
used JPEG compression (quality=50) on each frame.

### German traffic sign poisoned image classification

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

### Librispeech automatic speech recognition

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
    * At this time, the channel model consists only a single perfect acoustic channel. 
    Realistic channel models with arbitrary impulse responses may be implemented later, 
    at which time, the attack becomes "universal."
    * Black-box, white-box, and adaptive attacks will be performed on defenses.
  * Adversary Capabilities and Resources
    * To place an evaluation bound on the perceptibility of perturbations, the SNR is restricted to >20 dB.
* **Metrics of Interest:**
  * Primary metrics:
    * Word error rate, SNR
  * Derivative metrics - see end of document
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * (Primary) Targeted - [Imperceptible ASR attack](https://arxiv.org/abs/1903.10346) and Untargeted - 
  [Kenansville attack](https://arxiv.org/abs/1910.05262).
  * (Secondary) other "per-example" attacks such as PGD, FGM, may be considered for completeness.
* **Baseline Defense**: MP3 Compression
* **Baseline Model Performance: (results obtained using Armory v0.12.2)**
  * Baseline WER: 9.76% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 20dB) WER: 27.28% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 30dB) WER: 11.14% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 40dB) WER: 9.92% (1000 test examples)
  * Baseline Attack (Imperceptible ASR attack, max_iter_1st_stage = 100) WER: 62.54%, SNR: 30.45 dB (320 examples)
  * Baseline Attack (Imperceptible ASR attack, max_iter_1st_stage = 250) WER: 20.17%, SNR: 29.14 dB (320 examples)
  * Baseline Attack (Imperceptible ASR attack, max_iter_1st_stage = 400) WER: 11.36%, SNR: 29.48 dB (320 examples)
  * Baseline FGSM Attack (SNR = 20dB) WER: 30.78% (2620 test examples)
  * Baseline FGSM Attack (SNR = 30dB) WER: 20.87% (2620 test examples)
  * Baseline FGSM Attack (SNR = 40dB) WER: 16.07% (2620 test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.12.2)**\
Baseline defense is art_experimental.defences.mp3_compression_channelized()\
Baseline defense performance is evaluated for both black-box (untargeted) and white-box (targeted) attacks.
  * Baseline WER: 12.98% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 20dB) WER: 35.58% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 30dB) WER: 16.71% (1000 test examples)
  * Baseline Untargeted Attack (SNR = 40dB) WER: 13.73% (1000 test examples)
  * Baseline Attack (Imperceptible ASR attack) To be added
  * Baseline FGSM Attack (SNR = 20dB) WER: 33.16% (2620 test examples)
  * Baseline FGSM Attack (SNR = 30dB) WER: 23.32% (2620 test examples)
  * Baseline FGSM Attack (SNR = 40dB) WER: 18.62% (2620 test examples)

### so2sat multimodal image classification

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
    * Non-real time, digital evasion attack - the attack will be "universal" with respect to scaling, 
    rotation and translation.
    * Adversary may perturb a single modality (SAR or EO) or both modalities simultaneously (SAR and EO)
    * Black-box, white-box, and adaptive attacks will be performed on defenses.
  * Adversary Capabilities and Resources
    * Patch size < 20% of the image area
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), Patch size
  * Derivative metrics - see end of document 
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * (Primary) Adversarial Patch
  * (Secondary) other "per-example" attacks such as PGD, FGM, may also be considered for completeness.
* **Baseline Defense**: JPEG Compression for Multi-Channel
* **Baseline Model Performance: (results obtained using Armory v0.12.2)**
  * Baseline accuracy: 55.59% (all test examples)
  * Baseline accuracy: 58.30% (1000 test examples)
  * Baseline Attacked (Masked PGD, 5% patch, EO channels) accuracy: 0.5% (1000 test examples)
  * Baseline Attacked (Masked PGD, 10% patch, EO channels) accuracy: 0.0% (1000 test examples)
  * Baseline Attacked (Masked PGD, 15% patch, EO channels) accuracy: 0.0% (1000 test examples)
  * Baseline Attacked (Masked PGD, 5% patch, SAR channels) accuracy: 1.0% (1000 test examples)
  * Baseline Attacked (Masked PGD, 10% patch, SAR channels) accuracy: 2.7% (1000 test examples)
  * Baseline Attacked (Masked PGD, 15% patch, SAR channels) accuracy: 0.7% (1000 test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.12.2)**\
Baseline defense is art_experimental.defences.jpeg_compression_multichannel_image(clip_values=(0.0, 1.0),
quality=95, channel_first=False, apply_fit=False, apply_predict=True,
mins=[-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,], ranges=[2,2,2,2,1,1,1,1,1,1,1,1,1,1]).\
Baseline defense performance is evaluated for a white-box attack.
  * Baseline accuracy: 33.90% (1000 test examples)
  * Baseline Attacked (Masked PGD, 5% patch, EO channels) accuracy: 0.9% (1000 test examples)
  * Baseline Attacked (Masked PGD, 10% patch, EO channels) accuracy: 0.0% (1000 test examples)
  * Baseline Attacked (Masked PGD, 15% patch, EO channels) accuracy: 0.0% (1000 test examples)
  * Baseline Attacked (Masked PGD, 5% patch, SAR channels) accuracy: 1.4% (1000 test examples)
  * Baseline Attacked (Masked PGD, 10% patch, SAR channels) accuracy: 1.7% (1000 test examples)
  * Baseline Attacked (Masked PGD, 15% patch, SAR channels) accuracy: 0.3% (1000 test examples)

### xView object detection

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
    * Non-real time, digital evasion attack - attack will be "universal" with respect to scaling, rotation, 
    and translation.
    * Black-box, white-box, and adaptive attacks will be performed on defenses.
* Adversary Capabilities and Resources
    * Patch size <100x100 pixels
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of ground truth classes, Patch Size
  * Derivative metrics - see end of document 
  * Additional metrics specific to the scenario or that are informative may be added later
* **Baseline Attacks:**
  * (Primary) Masked PGD
  * (Secondary) other "per-example" attacks such as PGD, FGM, may be considered.
* **Baseline Defense**: JPEG Compression
* **Baseline Model Performance: (results obtained using Armory v0.12)**
  * Baseline mAP: 27.53% (all test examples)
  * Baseline mAP: 26.79% (1000 test examples)
  * Baseline Attacked (Masked PGD, 50x50 patch) mAP: 9.20% (1000 test examples)
  * Baseline Attacked (Masked PGD, 75x75 patch) mAP: 6.89% (1000 test examples)
  * Baseline Attacked (Masked PGD, 100x100 patch) mAP: 6.56% (1000 test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.12)**\
Baseline defense is art_experimental.defences.JpegCompressionNormalized(clip_values=(0.0, 1.0), quality=50,
channel_index=3, apply_fit=False, apply_predict=True).\
Baseline defense performance is evaluated for a white-box attack.
  * Baseline mAP: 21.75% (1000 test examples)
  * Baseline Attacked (Masked PGD, 50x50 patch) mAP: 11.89% (1000 test examples)
  * Baseline Attacked (Masked PGD, 75x75 patch) mAP: 9.95% (1000 test examples)
  * Baseline Attacked (Masked PGD, 100x100 patch) mAP: 9.03% (1000 test examples)

### APRICOT object detection

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
* **Baseline Model Performance: (results obtained using Armory v0.12.2)**
  * Baseline MSCOCO Objects mAP: 8.38% (all test examples)
  * Baseline Targeted Patch mAP: 5.52% (all test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.12.2)**\
Baseline defense is art_experimental.defences.jpeg_compression_normalized(clip_values=(0.0, 1.0), quality=10,
channel_index=3, apply_fit=False, apply_predict=True).\
Baseline defense performance is evaluated for a transfer attack.
  * Baseline MSCOCO Objects mAP: 7.38% (all test examples)
  * Baseline Targeted Patch mAP: 4.52% (all test examples)

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
