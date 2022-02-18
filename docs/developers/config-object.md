# an Armory Configuration object

As of armory 0.14.x application configuration is built up from various sources
and bits of it are sprinkled through the armory application. This results in
application state distributed hodge-podge through the app. Because the app
configuration is built piece-wise by multiple modules, proper unit test rigging
is difficult or impossible without hoisting the whole system up first.

This note outlines a configuration object to contain and render all application
configuration state. The state object is unrelated to armory "configurations",
represented by JSON blobs. The old configurations are now to be called "experiments"
because it is more descriptive.

## the ConfigurationTin class

There will be one object that contains all configuration data. For lack of
a better name at the moment, I'm calling it a ConfigurationTin.

    @dataclass
    class ConfigurationTin:
        mode: str
        flag: ArmoryFlags
        credential: ArmoryCredentials
        path: ArmoryPaths

Typical use is like

    from armory.configuration import tin

    tin = ConfigurationTin(…)

    if tin.mode == 'docker':
        docker_mount(tin.path.output_dir)
    ...
    armory.eval.run(tin, experiment)

But a test should be able to construct a ConfigurationTin from scratch like

    def test_with_overrides():
        my_tin = ConfigurationTin(overrides of some kind)
        armory.eval.run(my_tin, experiment)

## immutability declined

Python dataclasses can be marked as `frozen` but I'm not doing that to start. The armory
code currently modifies configuration all throughout which is a bad thing. When
converting to ConfigurationTin, we will know where all mutation happens because we will
have written it.

By getting the configuration modification well constructed, and having obvious
means of reading it, I expect the developer temptation to alter the tin will be
greatly reduced, and if it sneaks in, it will be much more obvious in review.
As the python maxim goes "we're all adults here"

## merging ConfigurationTins

We want the hierarchical override of configuration items drawn from:

    1. armory defaults
    2. overrides in the experiment file (aka config.json)
    2. overrides from environment variables
    4. override with command line arguments

As an example num-eval-batches is an experiment parameter that a user wants to
modify.

This mechanism does (not yet) address the hierarchical construction of configuration
which we think we want. If we used a json-like "bag of properties" as a `Dict[str, Any]`
we could use `dict.update` to trivially implement overrides. This desire does contain
a presupposition that we'd have a dictionary of modifiers to merge in. In actuality,
we'd have to construct that set of modifiers first, so why not use the
dataclass constructors for that?

There could be a method that handles this example

    args = argparse.parse()
    override: ArmoryFlags = tin.flag.copy()
    override.flag.skip_attack = args.skip_attack
    tin.meld(flag=override)

But I don't know that buys us anything over:

    tin.flag.skip_attack = args.skip_attack


## stuff that I read on the matter

https://dxiaochuan.medium.com/summary-of-python-config-626f2d5f6041 which
mentions https://github.com/apache/airflow/blob/175a1604638016b0a663711cc584496c2fdcd828/airflow/configuration.py#L233
as an exemplar
