# Dapricot Baseline Evaluation

Results obtained using Armory v0.13.3 and [dev test data](https://github.com/twosixlabs/armory/blob/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/armory/data/adversarial/dapricot_test.py)

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