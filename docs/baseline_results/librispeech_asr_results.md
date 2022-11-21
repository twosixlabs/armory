# Audio ASR Baseline Evaluation: 


## Deep Speech 2

Table 1 (Results obtained using Armory v0.13.3)

| Attack                   | Targeted | Budget         | Benign WER (Undefended) | Adversarial WER (Undefended) | Benign WER (Defended) | Adversarial WER (Defended) | Test Size |
|--------------------------|----------|----------------|-------------------------|------------------------------|-----------------------|----------------------------|-----------|
| Imperceptible ASR        | yes      | max_iter_1=100 | 0.10                    | 0.63                         | 0.13                  | N/A*                       | 320       |
| Imperceptible ASR        | yes      | max_iter_1=200 | 0.10                    | 0.20                         | 0.13                  | N/A                        | 320       |
| Imperceptible ASR        | yes      | max_iter_1=400 | 0.10                    | 0.11                         | 0.13                  | N/A                        | 320       |
| Kenansville              | no       | snr=20dB       | 0.10                    | 0.27                         | 0.13                  | 0.36                       | 1000      |
| Kenansville              | no       | snr=30dB       | 0.10                    | 0.11                         | 0.13                  | 0.17                       | 1000      |
| Kenansville              | no       | snr=40dB       | 0.10                    | 0.10                         | 0.13                  | 0.13                       | 1000      |
| PGD (single channel)     | no       | snr=20dB       | 0.10                    | 0.46                         | 0.13                  | 0.53                       | 100       |
| PGD (single channel)     | no       | snr=30dB       | 0.10                    | 0.46                         | 0.13                  | 0.50                       | 100       |
| PGD (single channel)     | no       | snr=40dB       | 0.10                    | 0.33                         | 0.13                  | 0.36                       | 100       |
| PGD (single channel)*    | yes      | snr=20dB       | 0.11                    | 1.03                         | 0.15                  | 1.01                       | 100       |
| PGD (single channel)*    | yes      | snr=30dB       | 0.11                    | 1.02                         | 0.15                  | 0.99                       | 100       |
| PGD (single channel)*    | yes      | snr=40dB       | 0.11                    | 0.88                         | 0.15                  | 0.84                       | 100       |
| PGD (multiple channels)  | no       | snr=20dB       | 0.13                    | 0.96                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)  | no       | snr=30dB       | 0.13                    | 0.59                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)  | no       | snr=40dB       | 0.13                    | 0.38                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=20dB       | 0.13                    | 0.99                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=30dB       | 0.13                    | 0.92                         | N/A                   | N/A                        | 100       |
| PGD (multiple channels)* | yes      | snr=40dB       | 0.13                    | 0.75                         | N/A                   | N/A                        | 100       |

* \*Targeted attack, where a random target phrase of similar length as the ground truth, was applied but WER wrt the ground truth was calculated

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/8eb10ac43bf4382d69625d8cef8a3e8cb23d0318/scenario_configs)
* Missing defended baseline is due to current incompatibility of the attack and defense.

Table 2 (Results are obtained using Armory v0.15.2)

| Attack | Targeted |  Budget  |      Attack Parameters      | Entailment/Contradiction/Neutral Rates (Benign Undefended) | Number of Entailment/Contradiction/Neutral Rates (Adversarial Undefended) | Entailment/Contradiction/Neutral Rates (Benign Defended) | Entailment/Contradiction/Neutral Rates (Adversarial Defended) | Test Size |
|:------:|:--------:|:--------:|:---------------------------:|:----------------------------------------------------------:|:-------------------------------------------------------------------------:|:--------------------------------------------------------:|:-------------------------------------------------------------:|:---------:|
| PGD*   | yes      | snr=20dB | eps_step=0.05, max_iter=500 | 0.95/0.05/0.00                                             | 0.01/0.98/0.01                                                            | 0.93/0.07/0.00                                           | 0.02/0.96/0.02                                                | 100       |
| PGD*   | yes      | snr=30dB | eps_step=0.03, max_iter=500 | 0.95/0.05/0.00                                             | 0.04/0.95/0.01                                                            | 0.93/0.07/0.00                                           | 0.19/0.79/0.02                                                | 100       |
| PGD*   | yes      | snr=40dB | eps_step=0.01, max_iter=500 | 0.95/0.05/0.00                                             | 0.43/0.53/0.04                                                            | 0.93/0.07/0.00                                           | 0.66/0.34/0.00                                                | 100       |

* \*Targeted attack, where contradictory target phrases are generated from ground truth phrases by changing a few key words (e.g., target phrase: `he is a bad person`; ground truth phrase: `he is a good person`)

Find reference baseline configurations [here](https://github.com/twosixlabs/armory/tree/master/scenario_configs/eval5/asr_librispeech)


## HuBERT

Coming soon