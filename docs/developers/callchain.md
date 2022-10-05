# bootstrap process creation and command line arguments

## Command Trace
This traces how command options percolate through armory instantiation

`armory.__main__.py` is the entry point for armory run. It has an `if __name__` block on
line 322 which calls main(). main() looks at only the first argument given (e.g. armory
**run**) and uses `run` as a lookup into a dispatch table COMMANDS which maps "run" ->
function `run`. run at line 284 does a bunch of argparse on the residual arguments,
loads the experiment config, constructs an Evaluator and then calls its run method.

`armory.eval.evaluator.Evaluator __init__` modifies the in-core experiment, sets up a
docker_client and other miscellany. the Evaluator.run method does some more prep and
then calls Evaluator.run_config which conses up a python command line with a base64
encoded experiment and then calls Evaluator.run_command which calls
armory.docker.management.exec_cmd which runs that encoded command inside a container.

That encoded command is `python -m armory.scenarios.main` which passes control via python
built-in hidden runpy.py which is currently complaining about import order in a way that
scares me:
> RuntimeWarning: 'armory.scenarios.main' found in sys.modules after import of
package 'armory.scenarios', but prior to execution of 'armory.scenarios.main'; this may
result in unpredictable behavior

In armory.scenarios.main in the `if __name__` block, first we have an independent
duplicate (and out of sync) argument processor which then calls main.run_config which
calls scenario.evaluate which finally runs application code.


## Environmental Variables
Armory currently utilizes 3 environmental variables to toggle on developer/testing features.
These variables can be set globaly- e.g. in an `rc` file or via the `export` command- or in
an ad hoc fashion on the command line like so, `ARMORY_DEV_MODE=1 armory run ...`

The current variables are as follow:

  - `ARMORY_DEV_MODE` (bool): Toggles developer mode💪 allowing the following variables to be read in.

  - `ARMORY_PRETEND_VERSION` (str): Overrides versioning mechanism with a custom version string. Useful when building containers locally.

  - `ARMORY_UPDATE_METADATA` (bool): Updates Armory's metadata in Python's site-package.

  - `ARMORY_BIND_MOUNT` (bool): Mounts your current working directory into the container.

### Example
```bash
ARMORY_DEV_MODE=1            \ # Toggle Developer Mode
ARMORY_PRETEND_VERSION=1.2.3 \ # Set the version to "1.2.3"
ARMORY_UPDATE_METADATA=1     \ # Update the pip version to "1.2.3"
ARMORY_BIN_MOUNT=1           \ # Mount `pwd` into the container
armory run                   \ # Off to the races!
  scenario_configs/eval6/poisoning/audio_backdoor_undefended.json \
  --use-gpu
```
