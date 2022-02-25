import armory
from armory.logs import log
import armory.logs
import click
from armory.eval import Evaluator
import sys


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


def execute_rig(config, root, interactive, jupyter_port, command):
    rig = Evaluator(config, root=root)
    exit_code = rig.run(
        interactive=interactive,
        jupyter=True if jupyter_port is not None else False,
        host_port=jupyter_port,
        command=command,
    )
    sys.exit(exit_code)


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
@click.option("--interactive", is_flag=True)
@click.argument("experiment")
def run(experiment, interactive):
    """Armory Run - Execute Armory using Experiment File

    EXPERIMENT - File containing experiment parameters
    """
    log.info(f"Executing Experiment: {experiment}")
    click.echo(f"Interactive: {interactive}")
    raise NotImplementedError("Still Working")
    # TODO Fix this


@cli.command()
@click.option("-d", "--default", is_flag=True, help="Use Defaults")
def setup(default):
    """Armory Setup - Setup the Armory Environment / Parameters
    """
    from armory.environment import setup_environment

    setup_environment()


@cli.command()
@click.option("--override", default=[], multiple=True)
def check(override):
    """Armory Check - Check basic Armory Setup
    """
    from armory.environment import EnvironmentParameters

    env = EnvironmentParameters.load(overrides=override)

    print(f"Armory Environment: \n {env}")


@cli.command()
@click.argument("experiment")
def validate(experiment):
    """Armory Check - Check basic Armory Setup
    """
    from armory.experiment import Experiment

    exp = Experiment.load(experiment)
    print(f"Armory Experiment: \n {exp}\n\n is valid!!")


@cli.command()
@click.option("-d", "--default", is_flag=True, help="Use Defaults")
def clean(default):
    """Armory Clean - Setup the Armory Environment / Parameters
    """
    raise NotImplementedError("Still Working")
    # TODO Update this to do the old `clean` bits


@cli.command()
def download():
    """Armory Download - Use Armory to Download things
    """
    raise NotImplementedError("Still Working")
    # TODO Implement this


@cli.command()
@click.argument("docker-image", type=str)
@click.option(
    "--interactive", is_flag=True, help="Allow Interactive Access to container"
)
@click.option(
    "-j",
    "--jupyter-port",
    default=None,
    type=click.IntRange(1, 65535),
    help="Specify Jupyter Port to use",
)
@docker_options
def launch(docker_image, interactive, jupyter_port, gpus, root):
    print(f"{jupyter_port}")
    execute_rig(
        config={
            "sysconfig": {
                "docker_image": docker_image,
                "use_gpu": True if gpus != "none" else False,
                "gpus": gpus,
            }
        },
        root=root,
        interactive=interactive,
        jupyter_port=jupyter_port,
        command="true # No-op",
    )


@cli.command()
@click.argument("docker-image", type=str)
@click.argument("command", type=str)
@docker_options
def exec(docker_image, command, gpus, root):
    """Armory Run - Execute Armory using Experiment File

    EXPERIMENT - File containing experiment parameters
    """
    log.info(
        f"Armory Executing command `{command}` using Docker Image: {docker_image} using "
        f"gpus: {gpus} and root: {root}"
    )

    execute_rig(
        config={
            "sysconfig": {
                "docker_image": docker_image,
                "use_gpu": True if gpus != "none" else False,
                "gpus": gpus,
            }
        },
        root=root,
        interactive=False,
        jupyter_port=None,
        command=command,
    )


if __name__ == "__main__":
    cli()
