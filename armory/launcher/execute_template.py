"""Execution Script for Armory Runs
"""
import time
$armory_sys_path # fmt: skip

import armory
import armory.logs
from armory.logs import log
import inspect
from armory.utils.environment import EnvironmentParameters
from armory.utils.experiment import ExperimentParameters
from armory.utils.utils import set_overrides
import importlib

environment_filename = "$environment_filename"
experiment_filename = "$experiment_filename"
output_directory = "$output_directory"
log_filters = $armory_log_filters # fmt: skip
armory.logs.update_filters(log_filters)


log.info("Running Execution Script for Armory")
log.warning(f"Using Armory Src: {inspect.getfile(armory)}")
environment = EnvironmentParameters.load(profile=environment_filename)
log.debug(f"Loaded Environment: \n{environment.pretty_print()}")

experiment, env_overrides = ExperimentParameters.load(filename=experiment_filename)
log.debug(f"Loaded Experiment: {experiment_filename}: \n {experiment.pretty_print()}")
if len(env_overrides) > 0:
    log.warning(
        f"Applying Environment Overrides from Experiment File: {experiment_filename}"
    )
    set_overrides(environment, env_overrides)
    log.debug(f"New Environment: \n{environment.pretty_print()}")

config = experiment.as_old_config()
config["environment_parameters"] = environment.dict()

# Import here to avoid dependency tree in launcher
log.debug("Loading Scenario Module & fn")
module = importlib.import_module(experiment.scenario.module_name)
ScenarioClass = getattr(module, experiment.scenario.function_name)
log.trace(f"ScenarioClass Loaded: {ScenarioClass}")
log.debug("Constructing Scenario Class...")
if experiment.scenario.kwargs is not None:
    pars = experiment.scenario.kwargs.dict()
    pars.pop("__root__")
else:
    pars = {}

scenario = ScenarioClass(config, output_directory=output_directory, **pars)
log.trace(f"constructed scenario: {scenario}")
log.debug("Calling .evaluate()")
scenario.evaluate()
log.success("Evaluation Complete!!")

