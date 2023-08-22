# Poisoning

Updated June 2023

Armory supports a handful of specific poisoning threat models and attacks.  This document will first describe these, providing enough background for newcomers to get up to speed on what these attacks do.  Then, the peculiarities of the poisoning configs will be addressed, including lots of helpful information about Witches' Brew.  Finally, we will describe the poisoning-specific metrics.


## Threat Models

Armory supports several poisoning threat models: dirty-label backdoor, clean-label backdoor, Witches' Brew and Sleeper Agent (clean-label gradient matching), and Object Detection (dirty-label backdoor).  In a backdoor attack, an adversary adds a small trigger, or backdoor, to a small portion of the train set in order to gain control of the the model at test time.
The trigger is usually a small (but not imperceptible) image superposed on the data, and the adversary's goal is to force the model to misclassify test images that have the trigger applied.  Armory includes several trigger images under `utils/triggers/`.


In poisoning attacks, the term _source class_ refers to the label of the image(s) that the adversary hopes to misclassify.  In the case of a targeted attack, the _target class_ is the desired misclassification label.  Neither of these terms describes which class gets poisoned; that depends on the threat model.  All of Armory's poisoning scenarios perform targeted attacks.  For simplicity, most Armory scenarios assume a single source class (all images to misclassify are from the same class) and a single target class (all misclassifications are aiming for the same label).  The exception is Witches' Brew, which accepts images from arbitrary source classes that can all have distinct targets.


### Dirty-label backdoor

In a [Dirty-label Backdoor (DLBD) Attack](https://arxiv.org/abs/1708.06733), training images are chosen from the source class, have a trigger applied to them, and then have their labels flipped to the target class.  The model is then trained on this modified data.  The adversary's goal is that test images from the source class will be classified as `target` when the trigger is applied at test time.

#### Audio

The DLBD attack for audio is similar to that of image. The difference is that instead of the trigger being an image that is placed over the existing image, the trigger is a short audio clip that is mixed with the existing audio. Example configs for speech are [here](../scenario_configs/eval6/poisoning)
Current triggers include a whistle and clapping.

### Clean-label backdoor

In a [Clean-label Backdoor (CLBD) Attack](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf), the triggers are applied to target-class images during training.  The poisoned samples also undergo some imperceptible gradient-based modifications to weaken the natural features, thus strengthening the association of the trigger with the target class label.
At test time, the adversary applies the trigger to source-class images in order to get them to misclassify as `target`.



### Witches' brew

[Witches' Brew](https://arxiv.org/abs/2009.02276) is a clean-label attack that does not use a backdoor trigger.  The associated scenario is complex enough to warrant its own document, [Witches' Brew Poisoning](poisoning_witches_brew.md).


### Sleeper Agent

[Sleeper Agent](https://arxiv.org/abs/2106.08970) is a clean-label attack that applies $l_\infty$ bounded perturbations to a set of training images to embed a hidden trigger into the model that can be applied at inference time.
This threat model does not assume access to the target architecture, but instead trains a surrogate model to produce the perturbations.
This approach uses gradient alignment to optimize the perturbations for the trigger.


### BadDet Object Detection Poisoning
[BadDet](https://arxiv.org/pdf/2205.14497.pdf) Object Detection Poisoning comprises 4 separate dirty-label object detection attacks which are described in [poisoning_object_detection.md](poisoning_object_detection.md).


## Configuration files

The config format for poisoning is currently complex and disorganized.  The metrics section is ignored.  
Parameters for attacks and defenses may be scattered between the attack, defense, and adhoc sections.
There are fields that seem to be copied and pasted from config to config with no consideration of whether they are needed.

A key thing to be aware of is that for DLBD attacks, the amount of data to poison is set under `adhoc/fraction_poisoned` and refers to the fraction of the source class to poison, not the whole dataset.  However, for CLBD attacks, this parameter is set under `attack/kwargs/pp_poison`.  For witches' brew, it is again set under `adhoc/fraction_poisoned`, but this time it refers to the percentage of the entire dataset.  In the latter case, since only the target class is poisoned, `fraction_poisoned` will be clipped to the actual size of the target class.  If there are multiple triggers with different target classes, the amount of poison will be split between target classes.

The `adhoc` section of the config is where most of the configuration action happens for poisoning attacks.  Most of the fields under `adhoc` would belong better in other sections.  A deprecation of the `adhoc` section is on the horizon, but in the meantime, here's a brief description.

The `adhoc` section is where `source_class`, `target_class`, and `train_epochs` are set.  The fields `compute_fairness_metrics` and `explanatory_model` go together, because the explanatory model is used to compute the fairness metrics, as described in the next section.  If the defense is a filtering defense and is separate from the model, it can be turned off with `use_poison_filtering_defense:false`.  Dataset poisoning can be turned off by setting `poison_dataset:false`; this has been the de facto approach to testing 0% poison, because ART throws an error in some cases when fraction poisoned is set to 0.  A final flag to note is `fit_defense_classifier_outside_defense`; this pertains to filters or other defenses that are external to the model and defaults to `true`.  If the defense does not require a trained model to operate, you can save time by setting this to `false`, because even if no defense classifier is provided, it will automatically train a copy of the model under evaluation .

The remaining sections are fairly straightforward.  The `attack` section carries the parameters for the attack (those not specified under `adhoc`, that is), including the size, position, and blend of backdoor triggers if applicable.  These are described under ```kwargs/backdoor_kwargs``` and typically depend on the exact ART object being loaded to perform the attack.  Refer to ART for relevant documentation.

The `defense` section for the _perfect filter_ baseline defense merits some explanation.  Because a perfect filter requires knowledge of which data were poisoned, and this information is not available to defenses, the perfect filter is implemented directly in scenario code.  However, Armory config validation currently requires a value for `module` and `name` under the `defense` section: the baseline configs set these to `"null"` (the string) although any string will work because the scenario ignores those values if `perfect_filter:true` is present in `defense/kwargs`.


## Metrics

The four primary poisoning metrics are:
- `accuracy_on_benign_test_data_all_classes`
- `accuracy_on_benign_test_data_source_class`
- `accuracy_on_poisoned_test_data_all_classes`
- `attack_success_rate`

These are computed after the model is trained on a poisoned dataset.  First, all of the test data is evaluated with no poison.  This gives us the first two metrics.  Next, data from the source class is poisoned and evaluated again.  The third metric, `accuracy_on_poisoned_test_data_all_classes`, is the total classification accuracy when the source class is poisoned.  The fourth metric, `attack_success_rate`, only measures the percentage of source-class examples that are misclassified as `target`.  In a well-executed attack, all these metrics will have high values: the first indicating that we have an effective, well-trained model;
the second confirming that the model is accurate on the source class; the third showing that total accuracy does not decrease substantially when one class is poisoned, and the fourth demonstrating that we can induce targeted misclassification in the source class.


If the defense under evaluation is a filtering defense, Armory will report traditional statistics on the filter, such as true and false positives and F1 score.
We also log the number of samples poisoned, and the number and percentage of samples filtered out of each class.  All metrics are computed automatically without regard to the `metric` field of the configuration file.

### Fairness Metrics

The GARD poisoning group has come up with two new per-class metrics to assess the bias of models within subpopulations of each class.
In the following explanation, these will be referred to as Filter Bias and Model Bias.

Both of these metrics are measured over sub-class clusters of data.
To obtain these clusters, a pre-trained _explanatory model_ (distinct from the model under evaluation) produces activations for all the data, and each class is then partitioned into two subclasses based on a _silhouette score_ computed on the activations.  This clustering method is intended to reflect one possible concept of majority/minority, by grouping data together whose silhouette score is within a range deemed to be "normal", while all the other data are considered as outliers in some respect.

Once we have this partitioning of each class, we can compute some interesting metrics.  The primary test is Statistical Parity Difference (SPD), which measures the difference in the probability of some event between two sets.
For two mutually exclusive sets $A$ and $B$, let $X$ be the event we care about, and $\mathrm{P}_A(X)$ is the probability of $X$ occuring over $A$.  Then $SPD_X(A,B) := \mathrm{P}_A(X) - \mathrm{P}_B(X).$  Values of SPD range from -1 to 1, with values closer to 0 indicating less bias.

For the Model Bias metric, the event of interest is correct classification.  Let $C_1$ and $C_2$ be a partition of a single class $C$ (i.e. $C = C_1 \cup C_2$ but $C_1 \cap C_2 = \emptyset$ ), and let $m(C)$ measure the number of elements of $C$ classified correctly by the model.
Then the Model Bias metric computes

$SPD_m(C) = \frac{m(C_1)}{|C_1|} - \frac{m(C_2)}{|C_2|}$.


The Filter Bias metric is very similar, only the event of interest is removal from the dataset by the filter.  Let $f(C)$ be the number of elements of $C$ that are removed by the filter.  Then the Filter Bias metric computes

$SPD_f(C) = \frac{f(C_1)}{|C_1|} - \frac{f(C_2)}{|C_2|}$.


In addition to SPD, we can compute any number of interesting statistics on the contingency tables formed by the subclass populations and the binary attributes of model correctness and filter removal.  Currently, Armory also reports the $\chi^2$ $p$-value of the contingency table for each class.  The $\chi^2$ test measures the likeliness of the contingency table if we expected no difference in model or filter behavior on different subpopulations of data.  Values range from 0 to 1, with a _higher_ value indicating _less_ bias.



### Filter Perplexity

Another filter metric we currently report is _perplexity_.  While the previous fairness metrics assess bias within individual classes, perplexity gives a bigger picture of filter bias between all classes.
The intuition behind this metric is that an unbiased filter should behave the same on all unpoisoned data, so that if there are false positives, they should not overwhelmingly be from a single class, but be spread evenly among all classes.
Perplexity characterizes the difference between two distributions with values from 0 to 1,
with a higher value indicating that the two distributions are more similar (it is equal to 1 if the distributions are identical).  We compare the class distribution of false positives with the class distribution of clean data.


Let $p(C_i)$ be the fraction of all false positives with class label $i$, and let $q(C_i)$ be the fraction of all unpoisoned datapoints with class label $i$.  Note that both $p$ and $q$ include the (unpoisoned part of the) poisoned class, as it is possible to be biased toward that class just as to any other.  Perplexity is defined as the exponential of the KL divergence between the two distributions, 
$e^{\mathrm{KL}(p||q)}$, where

$\mathrm{KL}(p||q) = \sum_i{p(C_i)\log}{\frac{p(C_i)}{q(C_i)}}$.
 

## Avoiding Out-Of-Memory (OOM) Errors

The poisoning scenarios typically work with very large in-memory numpy arrays.
This can result in OOM errors in certain cases.

One place where this can happen is when calling `fit` on the model.
If the model is a TensorFlowV2Classifier (ART), then the `fit` method will try to convert the entire numpy array to a tf tensor (not batched), which can exceed GPU memory for smaller cards, especially for the audio poisoning scenario.
If this is the case, set the scenario kwarg `fit_generator` to `true` in the config:
```
    ...
    "scenario": {
        "kwargs": {
            "fit_generator": true
        },
        ...
```
This will wrap the numpy array in a generator, and send batched inputs into the ART model's `fit_generator` method.
This is less efficient and trains slower than the other method, which is why is not default, but will avoid the OOM error in this case.

When `fit_generator` is set to `true`, and there is an explanatory model present, it will get activations from the dataset in a batched manner, which avoids a similar OOM error.
