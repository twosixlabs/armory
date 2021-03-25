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

    # find truthy sysconfig specifications
    sysconf = config["sysconfig"]
    new_spec = {name: sysconf[name] for name in sysconf if sysconf[name]}

    # find truthy args specifications, overwriting config if present
    cmd = vars(args)
    new_args = {name: cmd[name] for name in cmd if cmd[name]}
    new_spec.update(new_args)

    # sysconfig gets updated with all truthy members of the prioritized union
    sysconf.update(new_spec)

    # new_args now gets the original namespace and all truthy members of the prioritized
    # union
    cmd.update(new_spec)
    new_args = argparse.Namespace(**cmd)

    return config, new_args
