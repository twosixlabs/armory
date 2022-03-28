In armory versions prior to tiga, the configuration file is the center of
armory processing. This json document, sometimes referred to as `config.json`
describes all the resources used to conduct an experiment. This note
explains the means and rationale for transforming configs into experiments.

# serialization syntax

The Experiment file uses YAML syntax in favor over the json configs for one
main reason: JSON does not allow comments. It is a benefit to
