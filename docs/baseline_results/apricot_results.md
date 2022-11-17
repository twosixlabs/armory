# APRICOT Object Detection Baseline Evaluation (Updated December 2020)

* **Baseline Model Performance: (results obtained using Armory v0.13)**
  * Baseline MSCOCO Objects mAP: 8.76% (all test examples)
  * Baseline Targeted Patch mAP: 5.70% (all test examples)
* **Baseline Defense Performance: (results obtained using Armory v0.13)**
Baseline defense is art_experimental.defences.jpeg_compression_normalized(clip_values=(0.0, 1.0), quality=10,
channel_index=3, apply_fit=False, apply_predict=True).\
Baseline defense performance is evaluated for a transfer attack.
  * Baseline MSCOCO Objects mAP: 7.83% (all test examples)
  * Baseline Targeted Patch mAP: 4.59% (all test examples)

