# Baseline Models
Armory has several baseline models available for use in evaluations. All of these 
models return an ART wrapped classifier for use with ART attacks and defenses.


### Pretrained Weights
Pretrained weights can be loaded in to the baseline models or custom models. This is 
achieved by specifying the name in the `weights_file` field of a model's config. 

When the model is loaded it will first try to load the file from the armory 
`saved_model_dir`. This enables you to place your own custom weights in that directory 
for loading. If the weights file is not found it'll then try to download the file from 
our S3 bucket. Files that are available in the armory S3 bucket are listed in the table 
below. 

If the `weights_file` is not found locally or in the S3 bucket an error will be 
returned. 


### Keras
The model files can be found in [armory/baseline_models/keras](../armory/baseline_models/keras). 

| Model   | S3 weight_files   | 
|:----------: | :-----------: | 
| Cifar10 CNN |  |  
| Densenet121 CNN | `densenet121_resisc45_v1.h5` , `densenet121_imagenet_v1.h5` |
| Inception_ResNet_v2 CNN | `inceptionresnetv2_imagenet_v1.h5` |
| Micronnet CNN |  |
| MNIST CNN | `undefended_mnist_5epochs.h5` |
| ResNet50 CNN | `resnet50_imagenet_v1.h5` |
| so2sat CNN | `multimodal_baseline_weights.h5` |


### PyTorch
The model files can be found in [armory/baseline_models/pytorch](../armory/baseline_models/pytorch)

| Model   |                S3 weight_files                | 
|:----------: |:---------------------------------------------:| 
| Cifar10 CNN |                                               |  
| DeepSpeech 2 |                                               |
| Sincnet CNN |         `sincnet_librispeech_v1.pth`          |
| MARS | `mars_ucf101_v1.pth` , `mars_kinetics_v1.pth` |
| ResNet50 CNN |          `resnet50_imagenet_v1.pth`           |
| MNIST CNN |        `undefended_mnist_5epochs.pth`         |
| xView Faster-RCNN |  `xview_model_state_dict_epoch_99_loss_0p67`  |
| CARLA Faster-RCNN (rgb)|         `carla_rgb_weights_eval5.pt`          |
| CARLA Faster-RCNN (depth)|        `carla_depth_weights_eval5.pt`         |
| CARLA Faster-RCNN (multimodal)|      `carla_multimodal_naive_weights.pt`      |
| CARLA GoTurn|           `pytorch_goturn.pth.tar`            |
| YOLOv3 | `darknet53.conv.74`         |

### TensorFlow 1
The model file can be found in [armory/baseline_models/tf_graph](../armory/baseline_models/tf_graph). 
The weights for this model are downloaded from the link listed below.

| Model   | TF Weights URL   | 
|:----------: | :-----------: | 
| MSCOCO Faster-RCNN | http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz |


### Preprocessing Functions
Preprocessing functions have been moved inside each model's forward pass. This is to allow each
model to receive as input the canonicalized form of a dataset.