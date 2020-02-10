# Datasets

The data module implements functionality to download external and internal datasets to
the armory repository and standardizes them across all evaluations.

### Available Datasets
* `mnist` : MNIST hand written digit image dataset
* `cifar10`: CIFAR 10 classes image dataset
* `digit`:  Audio dataset of spoken digits
* `imagenet_adversarial`: ILSVRC12 adversarial image dataset for ResNet50

### TF Data Loading
We load datasets using `tf.data` and convert the data to numpy arrays for ingestion in 
any framework. While not the most efficient strategy, this current implementation 
helps us standardize evaluations across all frameworks.

### In-Memory or Generator
At the moment we have all datasets return as in memory NumPy arrays. This was done for 
the initial release so all examples align with their ART counterparts. Going forward 
we plan to generators so that larger datasets can be processed.