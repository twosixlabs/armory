# RESISC-45 Image Classification Baseline Evaluation

* **Baseline Model Performance: (results obtained using Armory < v0.10)**
  * Baseline Clean Top-1 Accuracy: 93%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 6%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 23%
* **Baseline Defense Performance: (results obtained using Armory < v0.10)**
Baseline defense is art_experimental.defences.JpegCompressionNormalized(clip_values=(0.0, 1.0), quality=50, channel_index=3, apply_fit=False,
apply_predict=True, means=[0.36386173189316956, 0.38118692953271804, 0.33867067558870334], stds=[0.20350874, 0.18531173, 0.18472934]) - see
resisc45_baseline_densenet121_adversarial.json for example usage.
Baseline defense performance is evaluated for a grey-box attack: adversarial examples generated on undefended baseline model evaluated on defended model.
  * Baseline Clean Top-1 Accuracy: 92%
  * Baseline Attacked (Universal Perturbation) Top-1 Accuracy: 40%
  * Baseline Attacked (Universal Patch) Top-1 Accuracy: 21%