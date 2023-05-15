# Witches' Brew Poisoning

[Witches' Brew](https://arxiv.org/abs/2009.02276) is a clean-label attack but there is no backdoor trigger involved.  The adversary selects individual `source` images from the test set; these are the images that the adversary wants to misclassify as `target` and are called _triggers_, not to be confused with the backdoor trigger in DLBD and CLBD attacks.  The attack uses a gradient-matching algorithm to modify a portion of the train-set target class, such that the unmodified test-set triggers will be misclassified.


## Configuration file

### Trigger image specification

Witches' Brew requires a `source_class`, `target_class`, and `trigger_index`.  The field `target_class` is required, but either of the other two may be left `null`.  If `trigger_index` is `null`, trigger images will be chosen randomly from the source class.  If `source_class` is `null`, it will be inferred from the class labels of images at the provided trigger index. 

Witches' Brew seeks to misclassify individual images; each has to be specified explicitly.  If multiple triggers are desired, there are several equivalent ways to accomplish this.  Some examples will illustrate.  Suppose you want three trigger images from class 1, each with a target class of 0.  The following configurations are equivalent:

```
source_class: 1
target_class: 0
trigger_index: [null, null, null]

source_class: [1,1,1]
target_class: 0
trigger_index: null

source_class: 1
target_class: [0,0,0]
trigger_index: null

source_class: [1,1,1]
target_class: [0,0,0]
trigger_index: null

source_class: [1,1,1]
target_class: [0,0,0]
trigger_index: [null, null, null]
```
Similarly, you can request triggers from different source classes by doing something like this:
```
source_class: [1,2,3]
target_class: 0
trigger_index: null
```
(selects triggers randomly from classes 1, 2, and 3, each with a target of 0).

Or this:
```
source_class: [null, null, null]
target_class: [4,5,6]
trigger_index: [10,20,30]
```
(Uses images 10, 20, and 30 as triggers, whatever their source label, with targets of 4, 5, and 6 respectively.  Note that source and target class may not be the same.)


### Witches' Brew dataset saving and loading

Because generating poisoned data takes so much longer for Witches' Brew than for the backdoor attacks, Armory provides a means to save and load a poisoned dataset.  A filepath may be provided in the config under `attack/kwargs/data_filepath`.  If this path does not exist, Armory will generate the dataset and save it to that path.  If the path does exist, Armory will load it and check that it was generated consistent with what the current config is requesting, in terms of source, target, perturbation bound, and so forth.  If there are any discrepancies, a helpful error is raised.  If you are loading a pre-generated dataset, `source_class`, `target_class`, and `trigger_index` may all be null.  If you want to re-generate a poisoned dataset that already exists, you can delete the old one or rename it.  Alternatively, you may set `attack.kwargs.overwrite_presaved_data:true` within the config, but use caution: if you forget to reset it to `false`, or pass the config to someone else, it can take a lot of time to re-generate the poison.



## Witches' Brew Metrics

Because test-time data is not poisoned for the witches' brew attack, it doesn't make sense to use the four primary metrics described in [poisoning.md](poisoning.md).  Instead, we have these three:
- `attack_success_rate`
- `accuracy_on_trigger_images`
- `accuracy_on_non_trigger_images`

`attack_success_rate` is the percentage of trigger images which were classified as their respective target classes, while `accuracy_on_trigger_images` is the percentage of trigger images that were classified as their natural labels (source classes).  Similarly, `accuracy_on_non_trigger_images` is the classification accuracy on non-trigger images.

The fairness and filter metrics remain the same.
