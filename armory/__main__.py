"""
python run_evaluation.py <json_config>

Try:
python run_evaluation.py examples/mnist_fgm_all_epsilon.json

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""

import argparse
import json
import logging
import sys


def main():
    try:
        import coloredlogs

        from armory.eval import Evaluator
    except ImportError as e:
        module = e.name
        print(f"ERROR: cannot import '{module}'", file=sys.stderr)
        try:
            with open("requirements.txt") as f:
                requirements = f.read().splitlines()
        except OSError:
            print(f"ERROR: cannot locate 'requirements.txt'", file=sys.stderr)
            sys.exit(1)

        if module in requirements:
            print(f"    Please run: $ pip install -r requirements.txt", file=sys.stderr)
        else:
            print(f"ERROR: {module} not in requirements. Please submit bug report!!!")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run Armory from config file.")
    parser.add_argument(
        "filepath", metavar="<json_config>", type=str, help="json config file"
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_const",
        const=True,
        default=False,
        help="Whether to all interactive access to container",
    )
    args = parser.parse_args()

    coloredlogs.install(level=args.log_level)
    with open(args.filepath) as f:
        config = json.load(f)
    rig = Evaluator(config)
    if args.interactive:
        rig.run_interactive()
    else:
        rig.run_config()


def download_all_datasets():
    """
    Script to download all datasets and docker container for offline usage.
    """

    import coloredlogs

    from armory.docker.management import ManagementInstance

    coloredlogs.install(level=logging.INFO)

    manager = ManagementInstance()
    runner = manager.start_armory_instance()
    runner.exec_cmd("python -c 'from armory.data import data; data.download_all()'")
    manager.stop_armory_instance(runner)


if __name__ == "__main__":
    main()
