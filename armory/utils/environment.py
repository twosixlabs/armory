"""
Environment parameter names
"""
from pydantic import BaseModel
import os
import json
import inspect
import armory
from armory.utils import set_overrides
import signal

DEFAULT_ARMORY_DIRECTORY = os.path.expanduser("~/.armory")


class Credentials(BaseModel):
    """Armory Credentials"""

    git_token: str = None
    s3_token: str = None
    verify_ssl: bool = True


class Paths(BaseModel):
    """Paths needed for armory use"""

    dataset_directory: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "datasets")
    git_repository_directory: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "git")
    output_directory: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "outputs")
    saved_models_directory: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "saved_models")
    temporary_directory: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "tmp")

    def check(self):
        for k, v in self.dict().items():
            if not os.path.isdir(v):
                raise NotADirectoryError(
                    f"armory environment path {k}: {v} does not exist"
                )

    def items(self):
        return self.dict().items()

    def change_base(self, base_dir):
        for k, v in self.dict().items():
            new_v = os.path.join(base_dir, os.path.basename(v))
            setattr(self, k, str(new_v))


class EnvironmentParameters(BaseModel):
    """Armory Environment Configuration Context

    This Dataclass contains the environmental references
    necessary for armory execution.

    Notes:
        To load from dictionary `pars`:
            env = EnvironmentParameters.parse_obj(pars)
    """

    profile: str = os.path.join(DEFAULT_ARMORY_DIRECTORY, "profile")
    armory_source_directory: str = os.path.dirname(inspect.getfile(armory))

    credentials: Credentials = Credentials()
    paths: Paths = Paths()

    def check(self):
        self.paths.check()

    def pretty_print(self):
        return json.dumps(self.dict(), indent=2, sort_keys=True)

    @classmethod
    def load(cls, profile=None, overrides=[]):
        if profile is None:
            profile = cls().profile

        env = cls.parse_file(profile)
        set_overrides(env, overrides)
        env.check()
        import os

        os.environ["ARMORY_PROFILE"] = env.profile
        os.environ["ARMORY_SOURCE_DIRECTORY"] = env.armory_source_directory
        for k, v in env.paths.dict().items():
            os.environ[f"ARMORY_PATHS_{str(k).upper()}"] = str(v)
        for k, v in env.credentials.dict().items():
            os.environ[f"ARMORY_CREDS_{str(k).upper()}"] = str(v)

        return env

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(json.dumps(self.dict(), indent=2, sort_keys=True))


def ask_yes_no(prompt, msg):
    while True:
        if prompt:
            print(prompt)
        response = str(input(f"{msg} [y/n]: "))
        if response.lower() not in ["y", "n", ""]:
            continue
        break
    return response


def get_value(msg, default_value, type, choices=None):
    try:
        answer = str(input(f"{msg} [DEFAULT: `{default_value}`]: "))
    except EOFError:
        answer = ""
    if not answer:
        answer = default_value

    if type == "dir":
        # Formatting Directory
        answer = os.path.abspath(os.path.expanduser(answer))
        if not os.path.exists(answer):
            response = ask_yes_no(
                None, f"Directory: {answer} does not exist, would you like to create?"
            )
            if response in ["y", ""]:
                print(f"\tCreating New Directory `{answer}`!!")
                os.makedirs(answer)
        return answer
    elif type == "bool":
        try:
            answer = bool(answer)
        except Exception as e:
            print(f"Invalid Bool Option: {answer}")
            raise e
        return answer
    elif type == "str":
        if choices and str(answer).lower() not in choices:
            print(f"Invalid Choice: {answer}...Not in {choices}")
        answer = str(answer).lower()
        return answer
    elif type == "list":
        if isinstance(answer, list):
            return answer
        elif isinstance(answer, str):
            answer = str(answer).split(" ")
            return answer
        else:
            raise ValueError(f"Unknown List Type: {type(answer)}")
    else:
        raise Exception(
            f"get_value must have one of `is_dir`, `is_bool`, `is_string` specified, received {type}"
        )


def save_profile(env_dict, filename):
    if os.path.exists(filename):
        print(f"WARNING: Profile File: {filename} already exists...overwriting!!")
        response = ask_yes_no(None, "Are you sure?")
        if response not in ("y", ""):
            print("Skipping Save!!")
            return
    print(f"Saving Profile to: {filename}")

    # Attempting to Load (verification)
    try:
        tmpfile = f"{filename}.tmp"
        print(f"Attempting Write to: {tmpfile}")
        with open(tmpfile, "w") as f:
            f.write(json.dumps(env_dict, indent=2, sort_keys=True))
        print(f"Reloading Environment from File: {tmpfile} and checking...")
        venv = EnvironmentParameters.parse_file(tmpfile)
        venv.check()
    except Exception as e:
        print("ERROR: Something went wrong during save/reload!!")
        print(f"Removing {tmpfile}")
        os.remove(tmpfile)
        print(f"Did not Save profile to {filename}!!!")
        raise e

    os.remove(tmpfile)
    with open(filename, "w") as f:
        f.write(json.dumps(env_dict, indent=2, sort_keys=True))


def setup_environment(use_defaults=False):
    # Construct Defaults
    env = EnvironmentParameters()

    if use_defaults:
        return env

    def handler(signum, frame):
        print("\n\nCtrl-c was pressed...aborting operation!!\n")
        exit(1)

    signal.signal(signal.SIGINT, handler)

    while True:
        if os.path.isfile(env.profile):
            print("WARNING: this will overwrite existing armory profile.")
            print("    Press Ctrl-C to abort.")

        # TODO: Update language about `absolute paths` to be more clear
        instructions = "\n".join(
            [
                "\nSetting up Armory Profile",
                f'    This profile will be stored at "{env.profile}"',
                "",
                "Please enter desired value for the following parameters.",
                "    If left empty, the default parameter will be used.",
                "    If specifying a directory, an absolute path must be provided,",
                "    which may include usage of special characters( e.g. '~') that ",
                "    will be resolved by this configuration tool.",
                "",
                "If, at any time, you wish to stop, press Ctrl-C.",
                "Note: This will result in the loss of the values you have already selected",
            ]
        )
        print(instructions)

        # Query for armory source directory
        new_val = get_value(
            "armory_source_directory", env.armory_source_directory, "dir", choices=None
        )
        setattr(env, "armory_source_directory", new_val)

        # Query for Credentials
        for k, v in env.credentials.dict().items():
            new_val = get_value(f"credentials.{k}", v, "str", choices=None)
            setattr(env.credentials, k, new_val)

        # Query for Paths
        for k, v in env.paths.dict().items():
            new_val = get_value(f"paths.{k}", v, "dir", choices=None)
            setattr(env.paths, k, new_val)

        # Saving (if selected)
        prompt = "\nSetup Information Selected:\n"
        prompt += json.dumps(env.dict(), indent=2, sort_keys=True)
        prompt += "\n"
        response = ask_yes_no(prompt, "Save this Profile?")
        if response in ("y", ""):
            print(f"\nSaving Armory Setup to: {env.profile}")
            save_profile(env.dict(), env.profile)
            break
        else:
            print("Configuration Not Saved!!!")

    print("\n Armory Setup Complete!!\n")
