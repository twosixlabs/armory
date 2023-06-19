# Global Misclassification Attack

## Globe trigger

### Undefended

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.527 | 0.460 | - | - | - | - | - |
| 01 |  0.499 | 0.467 | 0.500 | 0.470 | 0.081 | 0.243 | 0.016 |
| 05 |  0.514 | 0.493 | 0.169 | 0.313 | 0.149 | 0.447 | 0.620 |
| 10 |  0.487 | 0.463 | 0.173 | 0.323 | 0.152 | 0.457 | 0.635 |
| 20 |  0.510 | 0.467 | 0.166 | 0.317 | 0.164 | 0.493 | 0.701 |
| 30 |  0.472 | 0.453 | 0.156 | 0.327 | 0.162 | 0.487 | 0.688 |


### Random Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.432 | 0.403 | - | - | - | - | - |
| 01 |  0.448 | 0.430 | 0.447 | 0.427 | 0.074 | 0.223 | 0.014 |
| 05 |  0.441 | 0.430 | 0.221 | 0.340 | 0.118 | 0.353 | 0.427 |
| 10 |  0.448 | 0.450 | 0.160 | 0.300 | 0.144 | 0.433 | 0.587 |
| 20 |  0.460 | 0.430 | 0.163 | 0.297 | 0.149 | 0.447 | 0.637 |
| 30 |  0.424 | 0.423 | 0.126 | 0.277 | 0.152 | 0.457 | 0.672 |


### Perfect Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.518 | 0.460 | - | - | - | - | - |
| 01 |  0.549 | 0.473 | 0.549 | 0.473 | 0.082 | 0.247 | 0.024 |
| 05 |  0.538 | 0.473 | 0.539 | 0.473 | 0.081 | 0.243 | 0.016 |
| 10 |  0.516 | 0.457 | 0.517 | 0.457 | 0.079 | 0.237 | 0.020 |
| 20 |  0.480 | 0.440 | 0.480 | 0.437 | 0.074 | 0.223 | 0.014 |
| 30 |  0.481 | 0.423 | 0.480 | 0.427 | 0.073 | 0.220 | 0.012 |




## Baby-on-board trigger

### Undefended

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.518 | 0.457 | - | - | - | - | - |
| 01 |  0.508 | 0.487 | 0.507 | 0.487 | 0.080 | 0.240 | 0.014 |
| 05 |  0.488 | 0.473 | 0.379 | 0.433 | 0.106 | 0.317 | 0.219 |
| 10 |  0.502 | 0.457 | 0.157 | 0.290 | 0.149 | 0.447 | 0.623 |
| 20 |  0.524 | 0.483 | 0.173 | 0.343 | 0.173 | 0.520 | 0.704 |
| 30 |  0.484 | 0.460 | 0.140 | 0.270 | 0.142 | 0.397 | 0.566 |


### Random Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.454 | 0.430 | - | - | - | - | - |
| 01 |  0.463 | 0.417 | 0.464 | 0.417 | 0.076 | 0.227 | 0.024 |
| 05 |  0.427 | 0.417 | 0.350 | 0.390 | 0.089 | 0.267 | 0.186 |
| 10 |  0.462 | 0.450 | 0.156 | 0.277 | 0.146 | 0.437 | 0.592 |
| 20 |  0.461 | 0.443 | 0.146 | 0.273 | 0.160 | 0.480 | 0.690 |
| 30 |  0.406 | 0.403 | 0.121 | 0.283 | 0.159 | 0.420 | 0.628 |


### Perfect Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.519 | 0.463 | - | - | - | - | - |
| 01 |  0.532 | 0.467 | 0.532 | 0.467 | 0.082 | 0.247 | 0.026 |
| 05 |  0.521 | 0.460 | 0.520 | 0.460 | 0.079 | 0.237 | 0.017 |
| 10 |  0.529 | 0.457 | 0.530 | 0.457 | 0.080 | 0.240 | 0.016 |
| 20 |  0.518 | 0.447 | 0.519 | 0.450 | 0.079 | 0.237 | 0.014 |
| 30 |  0.493 | 0.447 | 0.494 | 0.447 | 0.077 | 0.230 | 0.015 |





# Object Generation Attack

## Globe Trigger

### Undefended

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.529 | 0.463 | - | - | - | - | - |
| 01 |  0.524 | 0.493 | - | - | 0.499 | 0.457 | 0.709 |
| 05 |  0.514 | 0.473 | - | - | 0.519 | 0.533 | 0.915 |
| 10 |  0.534 | 0.487 | - | - | 0.533 | 0.560 | 0.931 |
| 20 |  0.529 | 0.477 | - | - | 0.542 | 0.553 | 0.925 |
| 30 |  0.534 | 0.477 | - | - | 0.531 | 0.547 | 0.925 |


### Random Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.459 | 0.430 | - | - | - | - | - |
| 01 |  0.470 | 0.443 | - | - | 0.426 | 0.417 | 0.666 |
| 05 |  0.487 | 0.447 | - | - | 0.466 | 0.513 | 0.933 |
| 10 |  0.440 | 0.423 | - | - | 0.444 | 0.520 | 0.919 |
| 20 |  0.430 | 0.397 | - | - | 0.450 | 0.490 | 0.942 |
| 30 |  0.507 | 0.453 | - | - | 0.510 | 0.530 | 0.929 |


### Perfect Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.530 | 0.470 | - | - | - | - | - |
| 01 |  0.526 | 0.463 | - | - | 0.449 | 0.227 | 0.001 |
| 05 |  0.516 | 0.450 | - | - | 0.437 | 0.220 | 0.002 |
| 10 |  0.520 | 0.453 | - | - | 0.430 | 0.217 | 0.001 |
| 20 |  0.492 | 0.447 | - | - | 0.409 | 0.213 | 0.000 |
| 30 |  0.456 | 0.440 | - | - | 0.379 | 0.213 | 0.000 |




## Baby-on-board trigger

### Undefended

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.506 | 0.463 | - | - | - | - | - |
| 01 |  0.524 | 0.457 | - | - | 0.496 | 0.500 | 0.840 |
| 05 |  0.537 | 0.470 | - | - | 0.526 | 0.547 | 0.930 |
| 10 |  0.527 | 0.467 | - | - | 0.537 | 0.553 | 0.940 |
| 20 |  0.527 | 0.457 | - | - | 0.556 | 0.570 | 0.934 |
| 30 |  0.548 | 0.487 | - | - | 0.559 | 0.573 | 0.935 |


### Random Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.452 | 0.433 | - | - | - | - | - |
| 01 |  0.440 | 0.423 | - | - | 0.413 | 0.427 | 0.856 |
| 05 |  0.456 | 0.437 | - | - | 0.453 | 0.523 | 0.919 |
| 10 |  0.494 | 0.423 | - | - | 0.474 | 0.507 | 0.932 |
| 20 |  0.456 | 0.433 | - | - | 0.462 | 0.507 | 0.939 |
| 30 |  0.499 | 0.457 | - | - | 0.516 | 0.530 | 0.930 |


### Perfect Filter

| Poison Percentage | Benign mAP | Benign AP target | Adv mAP - Clean labels | Adv AP target - Clean labels | Adv mAP - Adv labels | Adv AP target - Adv labels | Attack success rate |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 00 |  0.532 | 0.480 | - | - | - | - | - |
| 01 |  0.526 | 0.460 | - | - | 0.444 | 0.223 | 0.000 |
| 05 |  0.529 | 0.460 | - | - | 0.446 | 0.220 | 0.001 |
| 10 |  0.527 | 0.447 | - | - | 0.442 | 0.220 | 0.001 |
| 20 |  0.503 | 0.443 | - | - | 0.418 | 0.213 | 0.000 |
| 30 |  0.474 | 0.437 | - | - | 0.399 | 0.210 | 0.001 |




