"""
Handling of command line arguments and other configuration.
"""

import argparse


def merge_config_and_args(config, args):
    """
    Override members of config if specified as args. The config dict is mutated.
    Members in config are percolated into args to act as if they were specified.
    Members of args that are not in config are put there so that the output
    accurately records what was run. Returns a modified config and a newly
    created args.
    The precedence becomes defaults < config block < command args.
    """

    sysconfig = config["sysconfig"]

    # the truth table is complicated because they are actually tri-states:
    # undef, defined but falsy, defined and truthy
    #
    # config    args    action
    # u         u       nothing
    # f         u       nothing
    # d         u       args <- config
    # u         f       nothing
    # f         f       nothing
    # d         f       args <- config
    # u         d       config <- args
    # f         d       config <- args
    # d         d       config <- args

    # find truthy config specifications
    new_spec = {}
    for name in sysconfig:
        if sysconfig[name]:
            new_spec[name] = sysconfig[name]

    # find truthy args specifications, overwriting config if present
    specified = vars(args)
    for name in specified:
        if specified[name]:
            new_spec[name] = specified[name]

    # sysconfig gets updated with prioritized union
    sysconfig.update(new_spec)

    # new_args now gets the original namespace and all truthy members of the prioritized
    # union
    specified.update(new_spec)
    new_args = argparse.Namespace(**specified)

    return config, new_args
