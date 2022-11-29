# xView Object Detection Baseline Evaluation (Updated July 2021)

results obtained using Armory v0.13.3

|     Attack    | Patch Size | Benign mAP (Undefended) | Adversarial mAP (Undefended) | Benign mAP (Defended) | Adversarial mAP (Defended) | Test Size |
|:-------------:|:----------:|:-----------------------:|:----------------------------:|:---------------------:|:--------------------------:|:---------:|
| Masked PGD    | 50x50      | 0.284                   | 0.142                        | 0.232                 | 0.139                      | 100       |
| Masked PGD    | 75x75      | 0.284                   | 0.071                        | 0.232                 | 0.094                      | 100       |
| Masked PGD    | 100x100    | 0.284                   | 0.076                        | 0.232                 | 0.092                      | 100       |
| Robust DPatch | 50x50      | 0.284                   | 0.193                        | 0.232                 | 0.184                      | 100       |
| Robust DPatch | 75x75      | 0.284                   | 0.184                        | 0.232                 | 0.146                      | 100       |
| Robust DPatch | 100x100    | 0.284                   | 0.173                        | 0.232                 | 0.165                      | 100       |

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)
