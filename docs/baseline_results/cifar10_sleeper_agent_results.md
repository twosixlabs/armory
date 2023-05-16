# Cifar10 Sleeper Agent Baseline Evaluation

**All tables are the mean of 3 runs.**  Results obtained using Armory 0.16.6 February 2023

## trigger_10.png

### Undefended

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.739 | 0.809 | - | - |
| 01 |  0.745 | 0.800 | 0.731 | 0.043 |
| 05 |  0.736 | 0.787 | 0.717 | 0.073 |
| 10 |  0.742 | 0.776 | 0.720 | 0.159 |
| 20 |  0.729 | 0.745 | 0.699 | 0.285 |
| 30 |  0.739 | 0.775 | 0.679 | 0.669 |


### Random Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.718 | 0.777 | - | - |
| 01 |  0.711 | 0.757 | 0.703 | 0.032 |
| 05 |  0.711 | 0.778 | 0.695 | 0.071 |
| 10 |  0.698 | 0.783 | 0.685 | 0.076 |
| 20 |  0.711 | 0.782 | 0.684 | 0.238 |
| 30 |  0.698 | 0.781 | 0.661 | 0.368 |


### Perfect Filter

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.741 | 0.800 | - | - |
| 01 |  0.725 | 0.659 | 0.711 | 0.031 |
| 05 |  0.741 | 0.749 | 0.729 | 0.018 |
| 10 |  0.741 | 0.796 | 0.733 | 0.023 |
| 20 |  0.738 | 0.794 | 0.728 | 0.027 |
| 30 |  0.738 | 0.808 | 0.729 | 0.020 |


### Activation Clustering

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.648 | 0.701 | - | - |
| 01 |  0.654 | 0.726 | 0.653 | 0.023 |
| 05 |  0.648 | 0.742 | 0.640 | 0.058 |
| 10 |  0.654 | 0.716 | 0.645 | 0.079 |
| 20 |  0.667 | 0.702 | 0.651 | 0.148 |
| 30 |  0.643 | 0.661 | 0.616 | 0.211 |


### Spectral Signatures

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.668 | 0.715 | - | - |
| 01 |  0.672 | 0.712 | 0.668 | 0.042 |
| 05 |  0.667 | 0.705 | 0.655 | 0.112 |
| 10 |  0.663 | 0.604 | 0.649 | 0.190 |
| 20 |  0.666 | 0.677 | 0.645 | 0.267 |
| 30 |  0.665 | 0.704 | 0.639 | 0.342 |


### DP-Instahide

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.680 | 0.757 | - | - |
| 01 |  0.684 | 0.788 | 0.682 | 0.020 |
| 05 |  0.686 | 0.780 | 0.682 | 0.020 |
| 10 |  0.674 | 0.796 | 0.670 | 0.015 |
| 20 |  0.681 | 0.793 | 0.677 | 0.026 |
| 30 |  0.679 | 0.802 | 0.674 | 0.046 |




