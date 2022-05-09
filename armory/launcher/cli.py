import armory
from armory.logs import log
import armory.logs
import click

# from armory.engine.eval import Evaluator
# import sys
import os


def setup_log(verbose, log_level):
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


# def execute_rig(config, root, interactive, jupyter_port, command):
#     rig = Evaluator(config, root=root)
#     exit_code = rig.run(
#         interactive=interactive,
#         jupyter=True if jupyter_port is not None else False,
#         host_port=jupyter_port,
#         command=command,
#     )
#     sys.exit(exit_code)


def docker_options(function):
    function = click.option(
        "--gpus",
        type=str,
        default="none",
        help="Specify GPU(s) to use. For example '3', '1,5', 'all'.. (default: None)",
    )(function)
    function = click.option("--root", is_flag=True, help="Run Docker at `root` user")(
        function
    )
    return function


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


# @cli.command()
# @click.option("--override", default=[], multiple=True)
# def check(override):
#     """Armory Check - Check basic Armory Setup
#     This replaces `--check` method from old armory run
#     """
#     from armory.environment import EnvironmentParameters
#
#     env = EnvironmentParameters.load(overrides=override)
#
#     print(f"Armory Environment: \n {env.pretty_print()}")


@cli.command()
@click.option("-d", "--default", is_flag=True, help="Use Defaults")
def clean(default):
    """Armory Clean - Setup the Armory Environment / Parameters"""
    raise NotImplementedError("Still Working")
    # TODO Update this to do the old `clean` bits


@cli.command()
def download():
    """Armory Download - Use Armory to Download things"""
    raise NotImplementedError("Still Working")
    # TODO Implement this


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


# def get_mounts_from_paths(path_list):
#     import armory.launcher.launcher as al
#
#     mounts = []
#     for name, dir_path in path_list:
#         mounts.append(
#             al.DockerMount(
#                 type="bind",
#                 source=dir_path,
#                 target=f"/armory/{os.path.basename(dir_path)}",
#                 readonly=False,
#             )
#         )
#     return mounts
#


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
    from armory.utils.environment import EnvironmentParameters

    env = EnvironmentParameters.load(overrides=override)

    requested_mode = "docker" if image is not None else "native"
    if env.execution_mode != requested_mode:
        log.warning(
            f"Overriding Armory execution_mode setting: {env.execution_mode}... exec called with mode: {requested_mode}"
        )

    cmd = " ".join(command)
    if requested_mode == "native":
        log.info(f"Executing command in `native` mode:\n\t{cmd}")
        al.execute_native_cmd(cmd)
    else:
        log.info(f"Executing command in `docker` mode:\n\t{cmd}")

        mounts = get_mounts_from_paths(env.paths.items())

        al.execute_docker_cmd(image, cmd, mounts=mounts)


@cli.command()
@click.argument("experiment")
@click.option("--override", default=[], multiple=True)
def run(experiment, override):
    """Armory Run - Execute Armory using Experiment File

    EXPERIMENT - File containing experiment parameters
    """
    from armory.utils.environment import EnvironmentParameters
    from armory.utils.experiment import ExperimentParameters
    from armory.utils.utils import set_overrides
    import armory.launcher.launcher as al

    log.info(f"Running Experiment Defined by: {experiment}")

    env = EnvironmentParameters.load(overrides=override)
    log.debug(f"Loaded Environment: \n{env.pretty_print()}")

    exp, env_overrides = ExperimentParameters.load(experiment, overrides=override)
    log.debug(f"Loaded Experiment: {experiment}: \n {exp.pretty_print()}")

    if len(env_overrides) > 0:
        log.warning(
            f"Applying Environment Overrides from Experiment File: {experiment}"
        )
        set_overrides(env, env_overrides)
        log.debug(f"New Environment: \n{env.pretty_print()}")

    al.execute_experiment(exp, env)


# @cli.command()
# @click.argument("docker-image", type=str)
# @click.argument("command", type=str)
# @docker_options
# def exec(docker_image, command, gpus, root):
#     """Armory Run - Execute Armory using Experiment File
#
#     EXPERIMENT - File containing experiment parameters
#     """
#     log.info(
#         f"Armory Executing command `{command}` using Docker Image: {docker_image} using "
#         f"gpus: {gpus} and root: {root}"
#     )
#
#     execute_rig(
#         config={
#             "sysconfig": {
#                 "docker_image": docker_image,
#                 "use_gpu": True if gpus != "none" else False,
#                 "gpus": gpus,
#             }
#         },
#         root=root,
#         interactive=False,
#         jupyter_port=None,
#         command=command,
#     )


if __name__ == "__main__":
    cli()
