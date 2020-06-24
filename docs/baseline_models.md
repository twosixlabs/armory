# Baseline Models
Armory has several baseline models available for use in evaluations. All of these 
models return an ART wrapped classifier for use with ART attacks and defenses.


### Pretrained Weights
Pretrained weights can be loaded in to the baseline models or custom models. This is 
acheived by specifying the name in the `weights_file` field of a model's config. 

When the model is loaded it will first try to load the file from the armory 
`saved_model_dir`. This enables you to place your own custom weights in that directory 
for loading. If the weights file is not found it'll then try to download the file from 
our S3 bucket. Files that are available in the armory S3 bucket are listed in the table 
below. 

If the `weights_file` is not found locally or in the S3 bucket an S3 error will be 
returned. It is [on the roadmap](https://github.com/twosixlabs/armory/issues/440) 
to return a more informative error.


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


### PyTorch
The model files can be found in [armory/baseline_models/pytorch](../armory/baseline_models/pytorch)

| Model   | S3 weight_files   | 
|:----------: | :-----------: | 
| Cifar10 CNN |  |  
| Sincnet CNN | `sincnet_librispeech_v1.pth` |
| MARS | `mars_ucf101_v1.pth` , `mars_kinetics_v1.pth` |
| ResNet50 CNN | `resnet50_imagenet_v1.pth` |
| MNIST CNN | `undefended_mnist_5epochs.pth` |


### Preprocessing Functions
Each model has it's own preprocessing function defined in it's model file. This is to 
ensure that data is preprocessed specifically for that model architecture. If there is 
no preprocessing function define the Dataset generator will return the defaults as 
defined in our [dataset documentation](datasets.md)
