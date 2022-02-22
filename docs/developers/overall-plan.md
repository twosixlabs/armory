# the big honking primo feature

We developing larger-scale changes in the work-branch called `shenshaw26/primo`.
The goal is to migrate features into twosixlabs/armory as they are tested and
GARD developers are updated. Keeping it is a separate branch which keeps closely
_tracked to_ develop using frequent merges so that merges _into_ develop
can be done opportunistically.

Although the overarching design is being worked on, this outlines some of the
major directions for discussion and review. Most sections of this note will
grow over the development effort to become their own design documents

# the launcher

A large part of what armory does is hoist up an execution environment and
populate it with data and model imports.

There is a natural cleave point in `Evaluator._run_config` where all execution
is delegated to a new `python -m armory.scenarios.main` which should have all
setup pushed to the upstream side of that point and all compute engine
activities must be pushed below the cleave. Once this has been done, the
launcher code can then be pulled off into pure-python launcher which requires
no ML framework libraries (tf, torch) or other platform (e.g. docker) present
to install

Formalizing this cleave point and the structures passed across will dramatically
help test rig construction since they can be versionable dependency injectors
instead of state sprinkled throughout the file system, environment variables, and
command arguments.

## argument processing

There are two disjoint invocations of argparse in `__main__.py` and `scenarios/main.py`.
The latter is actually an RPC deserialize step, with a corresponding re-serialize
step nearby.

# the Experiment object

The activities of armory focus around the evaluation of a block of engine
parameters call the Experiment. The former versions talk (confusingly) about
the armory "config file" which is distinct from the `~/.armory/config` file which
can be serialized to/from JSON. There is now an Experiment which can be serialized
to YAML as a `.aexp` file.

# the Configuration Tin

The Experiment object represents an evaluation job description, the `ConfigurationTin`
tin represents the execution environment for an experiment.  It is called a tin
because `config` and its lexically close synonyms have been tainted by the old
Experiment. So armory's first task is to create the tin from the various sources.
Once the tin is populated, the resources it requires can be acquired and experiment
is evaluated in that context.

As with the Experiment we're moving from amorphous bags of JSON properties to
well structured Python classes. A clear example is:

    class ArmoryCredentials
        github_token: str
        s3_id: str
        s3_secret: str
