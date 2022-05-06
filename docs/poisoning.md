# Poisoning

A great variety of poisoning attacks exist in the literature.  The threat models are somewhat less standardized
than for evasion attacks, and in many cases the deliniation between "attack" and "threat model" is very fuzzy.





## Threat Models

There are currently three primary threat models handled by Armory.

In a backdoor attack, an adversary adds a small trigger, or backdoor, to a small portion of the train set 
in order to gain control of the the model at test time.
The trigger is usually a small (but not imperceptible) image superposed on the data, and the adversary's 
goal is to force the model to misclassify test images that have the trigger applied.


### Dirty-label backdoor

In a dirty-label backdoor attack, training images are chosen from a _source_ class, have a trigger applied to them,
and then have their labels flipped to the _target_ class.  The adversary's goal is that test images from the _source_
class will be classified as _target_ class when the trigger is applied at test time.


### Clean-label backdoor

In a clean-label backdoor attack, the triggers are applied to _target_ class images during training.  The poisoned samples
may also undergo some imperceptible gradient-based modifications to weaken the natural features, thus strengthening the association
of the trigger with the target class label.
At test time, the adversary applies the trigger to _source_ class images in order to get them to misclassify as _target_.



### Witches' brew

Witches' brew is a clean label attack but there is no backdoor or trigger involved.  The adversary selects individual _source_ class images from 
the _test_ set; these are the images that the adversary wants to misclassify 
as _target_ and are called _triggers_.  The attack uses a gradient-matching algorithm to modify a portion of 
the _train_ set _target_ class, such that the unmodified test triggers will be misclassified.





## Configuration files






## Metrics