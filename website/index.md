# GARD — Guaranteeing AI Robustness Against Deception

The growing sophistication and ubiquity of machine learning (ML) components in advanced
systems dramatically expands capabilities, but also increases the potential for new
vulnerabilities. Current research on adversarial AI focuses on approaches where
imperceptible perturbations to ML inputs could deceive an ML classifier, altering its
response. Such results have initiated a rapidly proliferating field of research
characterized by ever more complex attacks that require progressively less knowledge
about the ML system being attacked, while proving increasingly strong against defensive
countermeasures. Although the field of adversarial AI is relatively young, dozens of
attacks and defenses have already been proposed, and at present a comprehensive
theoretical understanding of ML vulnerabilities is lacking.

The DARPA GARD program seeks to establish theoretical ML system foundations to identify
system vulnerabilities, characterize properties that will enhance system robustness, and
encourage the creation of effective defenses. Currently, ML defenses tend to be highly
specific and are effective only against particular attacks. GARD seeks to develop
defenses capable of defending against broad categories of attacks. Furthermore, current
evaluation paradigms of AI robustness often focus on simplistic measures that may not be
relevant to security. To verify relevance to security and wide applicability, defenses
generated under GARD will be measured in a novel testbed employing scenario-based
evaluations.

Two Six Technologies, IBM, MITRE, University of Chicago, and Google Research are
collaboratively generating platforms, libraries, datasets, and training materials to
holistically evaluate the robustness of AI models and defenses to adversarial attacks.


# Team — Holistic Evaluation of GARD-produced Defenses

![University of Chicago Logo][uchicago]
**TODO: @davidslater is there a blurb for UChicago?**

![Armory Logo][armory-logo]

[Armory][armory] is a platform for running repeatable,
scalable, robust evaluations of adversarial defenses. Configuration files are used to
launch local or cloud instances of the Armory docker containers. Models, datasets,
scenarios, and evaluation scripts can be pulled from external repositories or from the
baselines within this project. Armory strongly leverages ART library components for
attacks and model integration.

![Adversarial Robustness Toolbox Logo][art-logo]

[Adversarial Robustness Toolbox (ART)][art] is a Python library for Machine Learning Security.
ART provides tools that enable developers and researchers to defend and evaluate Machine
Learning models and applications against the adversarial threats of Evasion, Poisoning,
Extraction, and Inference. ART supports all popular machine learning frameworks
(TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy,
etc.), all data types (images, tables, audio, video, etc.) and machine learning tasks
(classification, object detection, speech recognition, generation, certification, etc.).



# APRICOT

![APRICOT Patch Example][patch]

MITRE is developing datasets and scenarios with which to robustly test relevant models
and defenses. One example of this is the public release of [APRICOT (Adversarial Patches
Rearranged in COnText)][apricot], a benchmark dataset created to enable reproducible
research on the real-world effectiveness of physical adversarial patch attacks on object
detection systems.



# Self-Study
## Pedagogical Materials for Learning to Evaluate Adversarial Robustness

The [Google Research SelfStudy repository][google] contains a collection of defenses aimed at researchers who wish to learn
how to properly evaluate the robustness of adversarial example defenses. While there is
a vast literature of published techniques that help to attack adversarial example
defenses, few researchers have practical experience actually running these. This project
is designed to give researchers that experience, so that when they develop their own
defenses, they can perform a thorough evaluation.


[uchicago]: images/uchicago.png
[armory-logo]: images/armory.png
[art-logo]: images/art.png
[patch]: images/patch.png
[armory]: https://github.com/twosixlabs/armory
[art]:  https://github.com/Trusted-AI/adversarial-robustness-toolbox
[apricot]: https://apricot.mitre.org/
[google]: https://github.com/google-research/selfstudy-adversarial-robustness
