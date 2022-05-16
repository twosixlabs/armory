import armory
from armory.logs import log
import armory.logs.styling as styles
import armory.logs
import click
import os


def get_id(root: bool):
    if os.name == "nt" or root:
        return "0:0"
    else:
        return f"{os.getuid()}:{os.getgid()}"


def setup_log(verbose, log_level):
    """Set the Armory Log levels given cli arguments

    Parameters:
        verbose (int):          Verbosity level (0=INFO, 1=DEBUG, 2=TRACE)
        log_level (list):       List of Log Levels to set
                                (e.g. ["armory:debug","matplotlib:warning"]
    """
    if len(log_level) > 0 and verbose:
        print(
            "Cannot Specify both `--verbose` and `--log-level`.  Please use one or the other"
        )
        exit()

    if len(log_level) > 0:
        print("Setting Log Levels using Filters: {}".format(log_level))
        armory.logs.update_filters(log_level)
    elif verbose:
        print("Setting Log Level using Verbose: {}".format(verbose))
        level = "DEBUG" if verbose == 1 else "TRACE"
        armory.logs.update_filters([f"armory:{level}"])
    else:
        print("Setting Log Level to Default")
        armory.logs.update_filters(["armory:INFO"])


def get_environment(overrides: list = []):
    """Get EnvironmentParameters Object given a set of overrides"""
    from armory.utils.environment import EnvironmentParameters
    log.debug("Constructing Environment")
    # If profile override is given, use that
    profile = [string.split("=")[1] for string in overrides if "profile" in string]
    profile = profile[0] if len(profile) == 1 else None
    if profile is not None:
        log.warning(f"Setting Environment from Profile: {profile}...set by override at command line")

    # Load the Environment
    env = EnvironmentParameters.load(profile=profile, overrides=overrides)
    log.trace(f"Loaded Environment: \n{env.pretty_print()}")
    return env

def docker_options(function):
    function = click.argument(
        "docker-image",
        type=str,
    )(function)
    function = click.option("--root", is_flag=True, help="Run Docker at `root` user")(
        function
    )
    return function

# # TODO: These don't seem like `docker` options... we should fix this.
# def docker_options(function):
#     """Helper method to encapsulate click options/arguments around docker execution"""
#     function = click.option(
#         "--gpus",
#         type=str,
#         default="none",
#         help="Specify GPU(s) to use. For example '3', '1,5', 'all'.. (default: None)",
#     )(function)
#     function = click.option("--root", is_flag=True, help="Run Docker at `root` user")(
#         function
#     )
#     return function


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("--log-level", default=[], multiple=True)
def cli(verbose, log_level):
    """
    ARMORY Adversarial Robustness Evaluation Test Bed provides
    a command line interface (CLI) to execute evaluations.

    For more details see: https://github.com/twosixlabs/armory
    For questions email us at: <armory@twosixlabs.com>
    """
    setup_log(verbose, log_level)


@cli.command()
@click.option("-d", "--default", is_flag=True, help="Use Defaults")
def setup(default):
    """Armory Setup - Setup the Armory Environment / Parameters"""
    from armory.utils.environment import setup_environment
    setup_environment()


@cli.command()
@click.argument(
    "command", type=str, nargs=-1
)  # trick from here: https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
@click.option(
    "-i",
    "--image",
    type=str,
    default=None,
    help="name of docker image to use (Default: None -- will execute in native mode",
)
@click.option("--override", default=[], multiple=True)
@docker_options
def exec(image, command, gpus, root, override):
    """Armory Launcher -- Execute command"""
    import armory.launcher.launcher as al
    env = get_environment(override)
    requested_mode = "docker" if image is not None else "native"

    # Format command for use in launcher
    cmd = " ".join(command).split(" ")
    if requested_mode == "native":
        log.info(f"Executing command in `native` mode:\n\t{cmd}")
        al.execute_cmd(cmd)
    else:
        log.info(f"Executing command in `docker` mode:\n\t{cmd}")
        mounts = al.get_mounts_from_environment(env)
        al.execute_docker_cmd(image, cmd, mounts=mounts)
    log.success("Execution Complete!!")


@cli.command()
@click.argument("experiment")
@click.option("--override", default=[], multiple=True)
def run(experiment, override):
    """Armory Run - Execute Armory Scenario specified by Experiment File

    Parameters:
        experiment: str             Filename of the Experiment File
        override: list              List of Overrides to apply
    """
    from armory.utils.environment import EnvironmentParameters
    from armory.utils.experiment import ExperimentParameters
    from armory.utils.utils import set_overrides
    import armory.launcher.launcher as al

    log.info(f"Running Experiment Defined by: {experiment}")
    # Load the initial environment
    env = get_environment(override)

    # Load the Experiment (get environment overrides specified in experiment file)]
    log.debug("Constructing Experiment")
    exp, env_overrides = ExperimentParameters.load(experiment, overrides=override)
    log.trace(f"Loaded Experiment: {experiment}: \n {exp.pretty_print()}")

    # Apply Environment Overrides (if applicable)
    if len(env_overrides) > 0:
        log.warning(
            f"Applying Environment Overrides from Experiment File: {experiment}"
        )
        set_overrides(env, env_overrides)
        log.debug(f"New Environment: \n{env.pretty_print()}")

    # Resetting Overrides as specified on command line (to keep precedence)
    set_overrides(env, override)
    log.debug(f"Experiment Specified:\n{exp.pretty_print()}\n\nEnvironment Specified: \n{env.pretty_print()}")

    # Executing the Experiment
    al.execute_experiment(exp, env)
    log.success("Run Complete!")

# TODO: Need to implement this once we settle on how to pass `check` flag information
#  to the ScenarioClass.  It seems there are two options:
#    1) Just pass the `check_flag`=True to the Scenario Constructor and let the
#       Scenario handle setting of the values.  Seems overly complicated and confusing.
#    2) Have either `cli.check` or `launcher.execute_experiment` manipulate the appropriate
#       parameters such that Scenario construction/exeuction is agnostic to whether it is a
#       check run or not.  This is preferred IMO (S. Henshaw)
# @cli.command()
# @click.argument("experiment")
# @click.option("--override", default=[], multiple=True)
# def check(experiment, override):
#     """Armory Run - Execute Armory Scenario specified by Experiment File
#
#     Parameters:
#         experiment: str             Filename of the Experiment File
#         override: list              List of Overrides to apply
#     """
#     from armory.utils.environment import EnvironmentParameters
#     from armory.utils.experiment import ExperimentParameters
#     from armory.utils.utils import set_overrides
#     import armory.launcher.launcher as al
#
#     log.info(f"Running Experiment Defined by: {experiment}")
#     # Load the initial environment
#     env = get_environment(override)
#
#     # Load the Experiment (get environment overrides specified in experiment file)]
#     log.debug("Constructing Experiment")
#     exp, env_overrides = ExperimentParameters.load(experiment, overrides=override)
#     log.trace(f"Loaded Experiment: {experiment}: \n {exp.pretty_print()}")
#
#     # Apply Environment Overrides (if applicable)
#     if len(env_overrides) > 0:
#         log.warning(
#             f"Applying Environment Overrides from Experiment File: {experiment}"
#         )
#         set_overrides(env, env_overrides)
#         log.debug(f"New Environment: \n{env.pretty_print()}")
#
#     # Resetting Overrides as specified on command line (to keep precedence)
#     set_overrides(env, override)
#     log.debug(f"Experiment Specified:\n{exp.pretty_print()}\n\nEnvironment Specified: \n{env.pretty_print()}")
#
#
#
#     # Executing the Experiment
#     al.execute_experiment(exp, env)
#     log.success("Run Complete!")

@cli.command()
@docker_options
@click.option("-jp","--jupyter-port",
              default=8888, type=click.IntRange(1, 65535),
              help="Specify the Jupyter Port to use on Host")
@click.option("--override", default=[], multiple=True)
def jupyter(docker_image, override, jupyter_port, root):
    import armory.launcher.launcher as al
    env = get_environment(override)
    command = [
        "jupyter",
        "lab",
        "--ip=0.0.0.0",
        f"--port={jupyter_port}",
        "--no-browser",
    ]
    if root:
        command.append("--allow-root")

    log.info(f"Executing command in `docker` mode:\n\t{' '.join(command)}")
    lines = [
        "About to launch jupyter.",
        styles.bold("# To connect on the command line as well, in a new terminal, run:"),
        # styles.bold(
        #     styles.red(
        #         f"docker exec -it -u {get_id(root)} {runner.docker_container.short_id} bash"
        #     )
        # ),
        # "",
        # styles.bold("# To run, inside of a notebook:"),
        # styles.bold(
        #     styles.red(
        #         "from armory.scenarios.main import get as get_scenario\n"
        #         f's = get_scenario("{docker_config_path}"{init_options}).load()\n'
        #         "s.evaluate()"
        #     )
        # ),
        # "",
        # styles.bold("# To gracefully shut down container, press: Ctrl-C"),
        # "",
        "Jupyter notebook log:",
    ]
    log.info("\n".join(lines))

    mounts = al.get_mounts_from_environment(env)
    ports = [al.DockerPort(host=jupyter_port,container=jupyter_port)]
    al.execute_docker_cmd(docker_image, command, mounts=mounts, ports=ports, cwd="/armory_src/")
    log.success("Execution Complete!!")



# @cli.command()
# @click.argument("docker-image", type=str)
# @click.option(
#     "--interactive", is_flag=True, help="Allow Interactive Access to container"
# )
# @click.option(
#     "-j",
#     "--jupyter-port",
#     default=None,
#     type=click.IntRange(1, 65535),
#     help="Specify Jupyter Port to use",
# )
# @docker_options
# def launch(docker_image, interactive, jupyter_port, gpus, root):
#     print(f"{jupyter_port}")
#     execute_rig(
#         config={
#             "sysconfig": {
#                 "docker_image": docker_image,
#                 "use_gpu": True if gpus != "none" else False,
#                 "gpus": gpus,
#             }
#         },
#         root=root,
#         interactive=interactive,
#         jupyter_port=jupyter_port,
#         command="true # No-op",
#     )




if __name__ == "__main__":
    cli()
