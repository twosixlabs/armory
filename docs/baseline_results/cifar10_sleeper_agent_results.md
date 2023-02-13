# Cifar10 Sleeper Agent Baseline Evaluation

Results obtained using Armory 0.16.4

### Undefended

Mean of 3 runs

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.735 | 0.740 | - | - |
| 01 |  0.739 | 0.770 | 0.726 | 0.038 |
| 05 |  0.738 | 0.771 | 0.722 | 0.135 |
| 10 |  0.739 | 0.788 | 0.715 | 0.212 |
| 20 |  0.743 | 0.780 | 0.698 | 0.524 |
| 30 |  0.731 | 0.794 | 0.670 | 0.753 |


### Random Filter

Mean of 3 runs

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.690 | 0.761 | - | - |
| 01 |  0.703 | 0.791 | 0.700 | 0.029 |
| 05 |  0.713 | 0.777 | 0.696 | 0.176 |
| 10 |  0.711 | 0.810 | 0.700 | 0.079 |
| 20 |  0.705 | 0.745 | 0.676 | 0.296 |
| 30 |  0.708 | 0.745 | 0.678 | 0.346 |


### Perfect Filter

Mean of 3 runs

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.749 | 0.800 | - | - |
| 01 |  0.727 | 0.694 | 0.715 | 0.045 |
| 05 |  0.741 | 0.749 | 0.729 | 0.018 |
| 10 |  0.741 | 0.767 | 0.731 | 0.028 |
| 20 |  0.731 | 0.778 | 0.725 | 0.009 |
| 30 |  0.741 | 0.807 | 0.736 | 0.013 |


### Activation Clustering

Mean of 3 runs

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.650 | 0.659 | - | - |
| 01 |  0.646 | 0.661 | 0.642 | 0.031 |
| 05 |  0.652 | 0.647 | 0.647 | 0.053 |
| 10 |  0.664 | 0.776 | 0.658 | 0.029 |
| 20 |  0.662 | 0.696 | 0.640 | 0.188 |
| 30 |  0.666 | 0.668 | 0.630 | 0.462 |


### Spectral Signatures

Mean of 3 runs

| Poison Percentage | Benign all classes | Benign source class | Adv. all classes | Attack success rate |
| ------- | ------- | ------- | ------- | ------- |
| 00 |  0.684 | 0.738 | - | - |
| 01 |  0.675 | 0.768 | 0.671 | 0.044 |
| 05 |  0.668 | 0.660 | 0.656 | 0.098 |
| 10 |  0.676 | 0.694 | 0.664 | 0.131 |
| 20 |  0.661 | 0.709 | 0.632 | 0.356 |
| 30 |  0.656 | 0.729 | 0.625 | 0.387 |

