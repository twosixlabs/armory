# Scenarios
Armory is intended to evaluate threat-model scenarios.
Baseline evaluation scenarios are described below.
Additionally, we've provided some academic standard scenarios.


## Configuration Files
Scenario configuration files are found in the `scenario_configs` directory [here](scenario_configs).
The most recent config files are found in the `eval6` subfolder and older configs are found in the `eval5` and `eval1-4` subfolders.
There are also symlinks to representative configs found in the base of the `scenario_configs` directory.


## Base Scenario Class
All scenarios inherit from the [Scenario](https://github.com/twosixlabs/armory/blob/master/armory/scenarios/scenario.py) class.
This class parses an armory configuration file and calls its `evaluate` method to perform all of the computation for a given threat-models robustness to attack.
All `evaluate` methods save a dictionary of recorded metrics which are saved into the armory `output_dir` upon  completion.
Scenarios are implemented as subclasses of `Scenario`, and typically given their own file in the [Scenarios Directory](https://github.com/twosixlabs/armory/blob/master/armory/scenarios/).

Of particular note is the [Poison](https://github.com/twosixlabs/armory/blob/master/armory/scenarios/poison.py) class, from which all poisoning scenarios are subclassed.
More information on poisoning scenarios is documented [here](poisoning.md).

### User Initialization

When adding custom metrics or instrumentation meters to a scenario, it may be necessary to initialize or perform user-specific operations before loading.
This can also be helpful for other goals, such as fine-grained control over random initializations, instantiating external integrations (e.g., TensorBoard), or setting things like environment variables.
For this purpose, there is a `user_init` method that is called at the beginning of `load` (but after scenario initialization).
In poisoning, this occurs right after random seed setting in `load` (to enable the user to easily override random initialization).

This uses the underlying scenario config field of the same name, `user_init`.
See [configuration](configuration_files.md) for the json specification.
An example config would be as follows:
```json
    ...
    "user_init": {
        "module": "import.path.to.my_module",
        "name": "my_init_function",
        "kwargs": {
             "case": 1,
             "print_stuff": false
        }
    }
}
```
Which would essentially do the following before loading anything else in the scenario:
```python
import import.path.to.my_module as my_module
my_module.my_init_function(case=1, print_stuff=False)
```
If `name` were `""` or `None`, then it would only do the import:
```python
import import.path.to.my_module
```

This could be helpful for a variety of things, such as registering `metrics` prior to loading or setting up custom meters.
For instance:
```python
def my_init_function():
    from armory.instrument import Meter, get_hub
    from armory import metrics
    m = Meter(
        "chi_squared_test",
        metrics.get("chi2_p_value"),
        "my_model.contingency_table",
    )
    get_hub().connect_meter(m)
```
Would enable measurement of a contingency table produced by your model.
This would require adding probe points in your model code to connect it (which doesn't need to be in the init block), e.g.:
```python
from armory.instrument import get_probe
probe = get_probe("my_model")

class MyModel(torch.nn.Module):
    ...
    def forward(x):
        ...
        table = np.array([[2, 3], [4, 6]])
        probe.update(contingency_table=table)
        ...
```


## Baseline Scenarios
Currently the following Scenarios are available within the armory package.
Some scenario files are tied to a specific attack, while others are customized for a given dataset.  Several are more general-purpose.
Along with each scenario description, we provide a link to a page with baseline results for applicable datasets and attacks.
More information about each referenced dataset can be found in the [datasets](datasets.md) document.


### Audio ASR (Updated June 2022)
* **Description:**
In this scenario, the system under evaluation is an automatic speech recognition system.
* **Dataset:**
  * Armory includes one dataset suited for ASR:
    * [LibriSpeech dataset](http://www.openslr.org/12) (custom subset)
* **Baseline Models:**
Armory includes two audio models:
  * [DeepSpeech 2](https://arxiv.org/pdf/1512.02595v1.pdf) with pretrained weights from either the AN4, LibriSpeech, or TEDLIUM datasets.  
  Custom weights may also be loaded by the model. *Deprecated: will be removed in version 0.17.0*
  * [HuBERT](https://arxiv.org/abs/2106.07447) Large from [torchaudio](https://pytorch.org/audio/0.10.0/pipelines.html#torchaudio.pipelines.Wav2Vec2Bundle)
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish for speech to be transcribed incorrectly
    * Targeted - an adversary may wish for specific strings to be predicted
    * Contradiction: an adversary may wish to transcribe a specific string with a meaning contrary to the original, albeit with a low word error rate.
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack.
    * Under some threat models, the channel model consists only a single perfect acoustic channel, and under others, it may consist of one additional multipath channel.
* **Metrics of Interest:**
  * Primary metrics:
    * Word error rate, SNR, entailment rate
  * Derivative metrics - see end of document
* **Baseline Attacks:**
  * [Imperceptible ASR attack](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr.py)
  * [PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Kenansville attack](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/attacks/kenansville_dft.py)  
* **Baseline Defense**: [MP3 Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/mp3_compression.py)
* **Baseline Evaluations:**
   * [LibriSpeech results](baseline_results/librispeech_asr_results.md)


### Audio Classification (Updated June 2020)
* **Description:**
In this scenario, the system under evaluation is a speaker identification system.
* **Dataset:**
  * Armory includes one dataset suited for Audio Classification:
    * [LibriSpeech dataset](http://www.openslr.org/12) (custom subset):
* **Baseline Model:**
  * Armory includes two baseline speaker classification models:
    * [SincNet](https://arxiv.org/abs/1808.00158), a scratch-trained model based on raw audio
    * A scratch-trained model based on spectrogram input (not mel-cepstrum or MFCC)
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to evade detection
    * Targeted - an adversary may wish to impersonate someone else
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack
    * Assuming perfect acoustic channel
    * Black-box, white-box, and adaptive attacks
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack computational cost, defense computational cost, various distance measures of perturbation
    (Lp-norms, Wasserstein distance, signal-to-noise ratio)
  * Derivative metrics - see end of document
* **Baseline Evaluations:**
  * [LibriSpeech results](baseline_results/librispeech_audio_classification_results.md)


### CARLA Multi-Object tracking (MOT) (Updated October 2022)
* **Description:**
In this scenario specific to the CARLA multi-object tracking dataset, the system under evaluation is an object tracker 
trained to track multiple pedestrians in video in an urban environment.
* **Dataset:**
The development dataset is the [CARLA](https://carla.org) Multi-Object Tracking dataset, with videos containing a green-screen in all frames intended for adversarial patch insertion.
The dataset contains natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed.
* **Baseline Model:**
  * Pretrained [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf) model with an [Faster-RCNN](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/carla_mot_frcnn_byte.py) base instead of Yolo.
* **Threat Scenario:**
  * Adversary objectives:
    * To degrade the performance of the tracker through the insertion of adversarial patches.
  * Adversary Operating Environment:
    * Non-real time, physical-like patch attacks
* Adversary Capabilities and Resources
    * Patch size of different size/shape as dictated by the green-screen in the frames. The adversary is expected to apply a patch with constant texture across all frames in the video, but the patch relative to the sensor may change due to sensor motion.
* **Metrics of Interest:**
  * Primary metrics are [HOTA](https://link.springer.com/article/10.1007/s11263-020-01375-2)-based (quotes taken from paper), taken from [TrackEval](https://github.com/JonathonLuiten/TrackEval) implementation.
    * mean DetA - "detection accuracy, DetA, is simply the percentage of aligning detections"
    * mean AssA - "association accuracy, AssA, is simply the average alignment between matched trajectories, averaged over all detections"
    * mean HOTA - "final HOTA score is the geometric mean of these two scores averaged over different localisation thresholds"
* **Baseline Attacks:**
  * [Custom Robust DPatch with Non-differentiable, Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/carla_obj_det_patch.py)
  * [Custom Adversarial Patch with Differentiable, Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/carla_obj_det_adversarial_patch.py)
* **Baseline Defense**: [JPEG Frame Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Evaluation**: [Carla MOT results](baseline_results/carla_mot_results.md)


### CARLA Multimodal Object Detection (Updated October 2022)
* **Description:**
In this scenario, the system under evaluation is an object detector trained to identify common objects in an urban environment.  This scenario handles multimodal data (RGB/depth).
* **Datasets** 
  The datasets are the [CARLA](https://carla.org) Object Detection and Overhead Object Detection datasets.
  These datasets contain natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed.
* **Baseline Model:**
  * Single-modality:
    * Pretrained [Faster-RCNN with ResNet-50](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/carla_single_modality_object_detection_frcnn.py) model.
  * Multimodal:
    * Pretrained multimodal [Faster-RCNN with ResNet-50](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/carla_multimodality_object_detection_frcnn.py) model.
* **Threat Scenario:**
  * Adversary objectives:
    * To degrade the performance of an object detector through the insertion of adversarial patches.
  * Adversary Operating Environment:
    * Non-real time, physical-like patch attacks
* Adversary Capabilities and Resources
    * Patch size of different size/shape as dictated by the green-screen in each image. In the multimodal case, both RGB and depth channels are to be perturbed.
* **Metrics of Interest:**
  * Primary metrics:
    * mAP
    * Disappearance rate
    * Hallucinations per image
    * Misclassification rate
    * True positive rate
* **Baseline Attacks:**
  * [Custom Robust DPatch with Non-differentiable, Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/art_experimental/attacks/carla_obj_det_patch.py)
  * [Custom Adversarial Patch with Differentiable, Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/art_experimental/attacks/carla_obj_det_adversarial_patch.py)
* **Baseline Defense**: [JPEG Compression](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/art_experimental/defences/jpeg_compression_normalized.py)
* **Baseline Evaluations**:
  * [Street-level dataset](baseline_results/carla_od_results.md#carla-street-level-od-dataset)
  * [Overhead dataset](baseline_results/carla_od_results.md#carla-overhead-od-dataset)


### CARLA Video Tracking (Updated July 2022)
* **Description:**
In this scenario, the system under evaluation is an object tracker trained to localize a single moving pedestrian.
* **Dataset:**
The development dataset is the [CARLA Video Tracking dataset](https://carla.org), which includes 20 videos, each of
which contains a green-screen in all frames intended for adversarial patch insertion. The dataset contains natural lighting metadata that allow digital, adaptive patches to be inserted and rendered into the scene similar to if they were physically printed.
* **Baseline Model:**
  * Pretrained [GoTurn](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/carla_goturn.py) model.
* **Threat Scenario:**
  * Adversary objectives:
    * To degrade the performance of the tracker through the insertion of adversarial patches.
  * Adversary Operating Environment:
    * Non-real time, physical-like patch attacks
* Adversary Capabilities and Resources
    * Patch size of different size/shape as dictated by the green-screen in the frames. The adversary is expected to apply a patch with constant texture across all frames in the video, but the patch relative to the sensor may change due to sensor motion.
* **Metrics of Interest:**
  * Primary metrics:
    * mean IOU
    * mean succss rate (mean IOUs are calculated for multiple IOU thresholds and averaged)
* **Baseline Attacks:**
  * [Custom Adversarial Texture with Input-Dependent Transformation](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/art_experimental/attacks/carla_adversarial_texture.py)
* **Baseline Defense**: [Video Compression](https://github.com/twosixlabs/armory/blob/v0.15.2/armory/art_experimental/defences/video_compression_normalized.py)
* **Baseline Evaluation**: [CARLA video tracking results](baseline_results/carla_video_tracking_results.md) 


### Dapricot Object Detection (Updated July 2021)
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
* Adversary Capabilities and Resources
    * Patch size of different shapes as dictated by the greenscreen sizes in the images
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of patches, Average target success
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Robust DPatch](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch_robust.py)
* **Baseline Defense:** [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Evaluation:** [Dapricot results](baseline_results/dapricot_results.md) 

### Image Classification
* **Description:**
In this scenario implements attacks against a basic image classification task.
* **Dataset:**
  * Armory includes several image classification datasets.
    * [Resisc-45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html).  It comprises 45 classes and 700 images for each class.  Images 1-500 of each class are in the training split,
500-600 are in the validation split, and 600-700 are in the test split.
    * MNIST
    * Cifar10
* **Baseline Models:**
  * Armory includes the following baseline image classification models:
    * Resisc-45: ImageNet-pretrained DenseNet-121 that is fine-tuned on RESISC-45.
    * MNIST: basic CNN
    * Cifar10: basic CNN
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to induce an arbitrary misclassification
    * Targeted - an adversary may wish to force misclassification to a particular class
  * Adversary Operating Environment:
    * Non real-time, digital evasion attack
    * Black-box, white-box, and adaptive attacks
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack computational cost, defense computational cost, various distance measures of perturbation
    (Lp-norms, Wasserstein distance)
  * Derivative metrics - see end of document
* **Baseline Defenses:** 
  * [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Evaluations:**
  * [Resisc-45 results](baseline_results/resisc45_results.md)


### Multimodal So2Sat Image Classification (Updated July 2021)
* **Description:**
In this scenario, the system under evaluation is an image classifier which determines local climate zone from a combination of co-registered synthetic aperture radar (SAR) and multispectral electro-optical (EO) images.  This Image Classification task gets its own scenario due to the unique features of the dataset.
* **Dataset:**
The dataset is the [so2sat dataset](https://mediatum.ub.tum.de/1454690). It comprises 352k/24k images in
train/validation datasets and 17 classes of local climate zones.
* **Baseline Model:**
  * Armory includes a custom CNN as a baseline model.  It has a single input that stacks SAR (first four channels only,
representing the real and imaginary components of the reflected electromagnetic waves)
and EO (all ten channels) data. Immediately after the input layer, the data is split into SAR and EO data
streams and fed into their respective feature extraction networks. In the final layer, the two
networks are fused to produce a single prediction output.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary wishes to evade correct classification
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack
    * Adversary perturbs a single modality (SAR or EO)
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), Patch size
  * Derivative metrics - see end of document
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
* **Baseline Defense:** [JPEG Compression for Multi-Channel](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/defences/jpeg_compression_multichannel_image.py)
* **Baseline Evaluation:** [So2Sat results](baseline_results/so2sat_results.md)  


### Object Detection
* **Description:**
In this scenario, the system under evaluation is an object detector.
* **Datasets:**
  * Armory includes two datasets for object detection (besides CARLA object detection which has its own [scenario](#carla-multimodal-object-detection-updated-october-2022)):
    * [xView](https://arxiv.org/pdf/1802.07856) comprises 59k/19k train and test
images (each with dimensions 300x300, 400x400 or 500x500) and 62 classes
    * [APRICOT](https://arxiv.org/pdf/1912.08166.pdf), which includes over 1000 natural images with physically-printed adversarial patches, with ten MS-COCO classes as targets
* **Baseline Models:**
   * [Faster-RCNN ResNet-50 FPN](https://arxiv.org/pdf/1506.01497.pdf), pre-trained, can be used for xView
   * [Faster-RCNN with ResNet-50, SSD with MobileNet, and RetinaNet](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) models, pretrained, can be used for APRICOT.
on MSCOCO objects and fine-tuned on xView.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary wishes to disable object detection
  * Adversary Operating Environment:
    * Non-real time, digital and physical-like evasion attacks
    and translation.
  * Note: the APRICOT dataset consists of advesarial images precomputed for a targeted attack.
* **Metrics of Interest:**
  * Primary metrics:
    * Average precision (mean, per-class) of ground truth classes, Patch Size
    * TIDE OD metrics
  * Derivative metrics - see end of document
* **Baseline Attacks:**
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Robust DPatch](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch_robust.py)
  * The patches for APRICOT were generated using variants of [ShapeShifter](https://arxiv.org/abs/1804.05810)
* **Baseline Defense:** [JPEG Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/jpeg_compression.py)
* **Baseline Evaluations**:
  * [xView results](baseline_results/xview_results.md)
  * [APRICOT results](baseline_results/apricot_results.md)

### UCF101 Video Classification

* **Description:**
In this scenario, the system under evaluation is a video action recognition system.
* **Datasets:**
Armory includes the following video classification datasets:
  * [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), which comprises 101 actions and 13,320 total videos. For the training/testing split,
we use the official Split 01.
* **Baseline Model:**
Armory includes a model for UCF101 that uses the [MARS architecture](http://openaccess.thecvf.com/content_CVPR_2019/papers/Crasto_MARS_Motion-Augmented_RGB_Stream_for_Action_Recognition_CVPR_2019_paper.pdf),
which is a single-stream (RGB) 3D convolution architecture that simultaneously mimics the optical flow stream.
The provided model is pre-trained on the Kinetics dataset and fine-tuned on UCF101.
* **Threat Scenario:**
  * Adversary objectives:
    * Untargeted - an adversary may simply wish to evade detection
  * Adversary Operating Environment:
    * Non-real time, digital evasion attack
* **Metrics of Interest:**
  * Primary metrics:
    * Accuracy (mean, per-class), attack budget
  * Derivative metrics - see end of document
* **Baseline Attacks:**
  * [Frame Saliency](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/frame_saliency.py)
  * [Masked PGD](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/projected_gradient_descent/projected_gradient_descent.py)
  * [Flicker Attack](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/over_the_air_flickering/over_the_air_flickering_pytorch.py)
  * [Custom Frame Border attack](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/art_experimental/attacks/video_frame_border.py)
* **Baseline Defense:** [Video Compression](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/preprocessor/video_compression.py)
* **Baseline Evaluations:**
  * [UCF101 results](baseline_results/ucf101_results.md)


### Poisoning

For a complete overview of the poisoning scenarios, threat models, attacks, and metrics, see the [poisoning doc](poisoning.md).  Here, we will briefly summarize each scenario and link to the baseline results.

#### Poison base scenario (DLBD)

* **Description:** The base scenario implements a Dirty-label Backdoor attack (DLBD).  In this scenario, the attacker is able to poison a percentage of the training data by adding backdoor triggers and flipping the label of data examples.  Then, the attacker adds the same trigger to test images to cue the desired misclassification.  For a complete overview, see the [poisoning doc](poisoning.md).
* **Datasets:**
  Datasets for DLBD include but are not limited to:
  * GTSRB
  * Audio Speech Commands
  * Resisc-10
  * Cifar10
* **Baseline Models:**
  Armory includes several models which may be used for this scenario:
  * [GTSRB micronnet](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/micronnet_gtsrb.py)
  * [Audio resnet](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/tf_graph/audio_resnet50.py)
  * [Resnet18](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/resnet18.py) can be used for Cifar10 or Resisc-10
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted misclassification
  * Adversary Operating Environment:
    * Digital dirty label poisoning attack
* **Metrics of Interest:** See the [poisoning doc](poisoning.md) for a full description of these metrics.
  * accuracy_on_benign_test_data_all_classes
  * accuracy_on_benign_test_data_source_class
  * accuracy_on_poisoned_test_data_all_classes
  * attack_success_rate
  * Model Bias fairness metric
  * Filter Bias fairness metric
* **Baseline Defenses:**
  * [Activation Clustering](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py)
  * [Spectral Signatures](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py)
  * [DP-InstaHide](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/trainer/dp_instahide_trainer.py)
  * [Random Filter](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/poison_detection/random_filter.py)
  * [Perfect Filter](https://github.com/twosixlabs/armory/blob/1d6caa9166313c1409edbbc5f089d2bc774b5230/armory/scenarios/poison.py#L233-L235)
* **Baseline Evaluations:**
  * [GTSRB DLBD](baseline_results/gtsrb_dlbd_results.md)
  * [Resisc DLBD](baseline_results/resisc_dlbd_results.md)
  * [Audio](baseline_results/speech_commands_poison_results.md) 
  * [Cifar10](baseline_results/cifar10_dlbd.md)


#### Poisoning CLBD
* **Description:** This scenario implements a Clean-label Backdoor attack (CLBD).  In this scenario, the attacker adds triggers to source class training images, leaving the labels the same but also applying imperceptible perturbations that look like target class features.  At test time, adding the trigger to a source class image induces misclassification to the target class.  For a complete overview, see the [poisoning doc](poisoning.md).
* **Datasets:**
  Datasets for CLBD include but are not limited to:
  * GTSRB
* **Baseline Models:**
  Armory includes several models which may be used for this scenario:
  * [GTSRB micronnet](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/micronnet_gtsrb.py)
  * [Resnet18](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/resnet18.py)
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted misclassification
  * Adversary Operating Environment:
    * Digital clean label poisoning attack
* **Metrics of Interest:** See the [poisoning doc](poisoning.md) for a full description of these metrics.
  * accuracy_on_benign_test_data_all_classes
  * accuracy_on_benign_test_data_source_class
  * accuracy_on_poisoned_test_data_all_classes
  * attack_success_rate
  * Model Bias fairness metric
  * Filter Bias fairness metric
* **Baseline Defenses:**
  * [Activation Clustering](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py)
  * [Spectral Signatures](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py)
  * [DP-InstaHide](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/trainer/dp_instahide_trainer.py)
  * [Random Filter](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/poison_detection/random_filter.py)
  * [Perfect Filter](https://github.com/twosixlabs/armory/blob/1d6caa9166313c1409edbbc5f089d2bc774b5230/armory/scenarios/poison.py#L233-L235)
* **Baseline Evaluations:**
  * [GTSRB](baseline_results/gtsrb_clbd_results.md)
  * [Resisc CLBD](baseline_results/resisc_clbd_results.md)


#### Poisoning: Sleeper Agent
* **Description:** This scenario implements the Sleeper Agent attack.  In this scenario, the attacker poisons train samples through gradient matching, then applies a trigger to test images to induce misclassification.
For a complete overview, see the [poisoning doc](poisoning.md).
* **Datasets:**
  Datasets for Sleeper Agent include but are not limited to:
  * Cifar10
* **Baseline Models:**
  Models include but are not limited to:
* [Resnet18](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/resnet18.py)
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted misclassification
  * Adversary Operating Environment:
    * Digital clean label poisoning attack
* **Metrics of Interest:** See the [poisoning doc](poisoning.md) for a full description of these metrics.
  * accuracy_on_benign_test_data_all_classes
  * accuracy_on_benign_test_data_source_class
  * accuracy_on_poisoned_test_data_all_classes
  * attack_success_rate
  * Model Bias fairness metric
  * Filter Bias fairness metric
* **Baseline Defenses:**
  * [Activation Clustering](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py)
  * [Spectral Signatures](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py)
  * [DP-InstaHide](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/trainer/dp_instahide_trainer.py)
  * [Random Filter](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/poison_detection/random_filter.py)
  * [Perfect Filter](https://github.com/twosixlabs/armory/blob/1d6caa9166313c1409edbbc5f089d2bc774b5230/armory/scenarios/poison.py#L233-L235)
* **Baseline Evaluations:**
  * [Cifar results](baseline_results/cifar10_sleeper_agent.md)


#### Poisoning: Witches' Brew
* **Description:** This scenario implements the Witches' Brew attack.  In this scenario, the attacker poisons train samples through gradient matching, to induce misclassification on a few individual pre-chosen test images.  For a complete overview, see the [witches' brew poisoning doc](poisoning_witches_brew.md).
* **Datasets:**
  The following datasets have been successfully used in this scenario:
  * GTSRB
  * Cifar10
* **Baseline Models:**
  Armory includes several models which may be used for this scenario:
  * [GTSRB micronnet](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/micronnet_gtsrb.py)
  * [Resnet18](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/resnet18.py)
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted misclassification
  * Adversary Operating Environment:
    * Digital clean label poisoning attack
* **Metrics of Interest:** See the [WB poisoning doc](poisoning_witches_brew.md) for a full description of these metrics.
  * accuracy_on_trigger_images
  * accuracy_on_non_trigger_images
  * attack_success_rate
  * Model Bias fairness metric
  * Filter Bias fairness metric
* **Baseline Defenses:**
  * [Activation Clustering](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py)
  * [Spectral Signatures](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py)
  * [DP-InstaHide](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/trainer/dp_instahide_trainer.py)
  * [Random Filter](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/poison_detection/random_filter.py)
  * [Perfect Filter](https://github.com/twosixlabs/armory/blob/1d6caa9166313c1409edbbc5f089d2bc774b5230/armory/scenarios/poison.py#L233-L235)
* **Baseline Evaluations:**
  * [Cifar10 results](baseline_results/cifar10_witches_brew_results.md)
  * [GTSRB results](baseline_results/gtsrb_witches_brew_results.md)


#### Poisoning: Object Detection
* **Description:** This scenario implements the four BadDet Object Detection Poisoning attacks: Regional Misclassification, Global Misclassification, Object Disappearance, and Object Generation.  For a complete overview, see the [object detection poisoning doc](poisoning_object_detection.md).
* **Datasets:**
  Datasets of interest include but are not limited to:
  * Minicoco
* **Baseline Models:**
  Object Detection models in Armory include but are not limited to:
  * [YOLOv3](https://github.com/twosixlabs/armory/blob/master/armory/baseline_models/pytorch/yolov3.py)
* **Threat Scenario:**
  * Adversary objectives:
    * Targeted misclassification (regional and global)
    * Targeted object generation
    * Targeted object disappearance
  * Adversary Operating Environment:
    * Digital dirty label poisoning attack
* **Metrics of Interest:** See the [OD poisoning doc](poisoning_object_detection.md) for a full description of these metrics.
  * AP on benign data
  * AP on adversarial data with benign labels
  * AP on adversarial data wtih adversarial labels
  * Attack success rate (misclassification, disappearance, generation)
  * TIDE OD metrics
* **Baseline Defenses:**
  * [Activation Clustering](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py)
  * [Spectral Signatures](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/spectral_signature_defense.py)
  * [DP-InstaHide](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/trainer/dp_instahide_trainer.py)
  * [Random Filter](https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/poison_detection/random_filter.py)
  * [Perfect Filter](https://github.com/twosixlabs/armory/blob/1d6caa9166313c1409edbbc5f089d2bc774b5230/armory/scenarios/poison.py#L233-L235)
* **Baseline Evaluations:**
  * [MiniCoco results](baseline_results/object_detection_poisoning_results.md)



## Creating a new scenario
Users may want to create their own scenario, because the baseline scenarios do
not fit the requirements of some defense/threat-model, or because it may be easier
to debug in code that you have access to as opposed to what is pre-installed by the
armory package.

To do so, simply inherit the scenario class and override the necessary functions.
An [example of doing this](https://github.com/twosixlabs/armory-example/blob/master/example_scenarios/audio_spectrogram_classification.py) can be found in our armory-examples repo.
