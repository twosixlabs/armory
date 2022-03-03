"""
Environment parameter names
"""
from pydantic import BaseModel, validator
import os
from armory.logs import log
from typing import List, Union


from enum import Enum, IntEnum


class ExecutionMode(str, Enum):
    """Armory Execution Mode
    docker  ->  Means that armory will execute
                experiments inside the prescribed
                docker container
    native  ->  Means that armory will execute
                experiments in the python environment
                in which it is called.
    """
    docker = 'docker'
    native = 'native'


class Credentials(BaseModel):
    """Armory Credentials"""
    # TODO: maybe have a `credential_file` ?
    git_token: str = None
    s3_token: str = None
    verify_ssl: bool = True


class Paths(BaseModel):
    """Paths to various armory items
    NOTE: At construction, the fiels are
    validated to make sure they exist.
    """
    setup_file: str = os.path.expanduser(os.path.join("~",".armory","profile"))
    dataset_directory: str = os.path.expanduser(os.path.join("~",".armory","datasets"))
    git_repository_directory: str = os.path.expanduser(os.path.join("~",".armory","git"))
    output_directory: str = os.path.expanduser(os.path.join("~",".armory","outputs"))
    saved_models_directory: str = os.path.expanduser(os.path.join("~",".armory","saved_models"))
    temporary_directory: str = os.path.expanduser(os.path.join("~",".armory","tmp"))

    # '*' is the same as 'cube_numbers', 'square_numbers' here:
    @validator('*', pre=True)
    def split_str(cls, v):
        if not os.path.exists(v):
            msg = f"armory environment path: {v} does not exist"
            raise ValueError(msg)
        return v

DOCKER_REPO = "twosixarmory"

# TODO Think aobut what to do if multiple simulatenous test
#  builds are happending on same machine with image name/tag
#  clashes


class DockerImage(str, Enum):
    """Armory Supported Docker Images

    IMAGES
    base:                   This image contains the base requirements
                            necessary to run armory within a container.
                            NOTE: This image does NOT contain an installation
                            of armory.

    tf2                     Image with armory installed and ready to use with
                            TensorFlow2 based architectures

    pytorch                 Image with armory installed and ready to use with
                            pytorch based architectures

    pytorch-deepspeech      Image with armory installed and ready to use with
                            pytorch deepspeech architectures (primarily used with
                            audio scenarios
    """
    base = f"{DOCKER_REPO}/base"
    pytorch = f"{DOCKER_REPO}/pytorch"
    pytorch_deepspeech = f"{DOCKER_REPO}/pytorch-deepspeech"
    tf2 = f"{DOCKER_REPO}/tf2"


class EnvironmentParameters(BaseModel):
    """Armory Environment Configuration Context

    This Dataclass contains the environmental references
    necessary for armory execution.
    """
    execution_mode: ExecutionMode = ExecutionMode.native
    credentials: Credentials
    paths: Paths
    armory_images: List[Union[str, DockerImage]] = [DockerImage.tf2,
                                                    DockerImage.pytorch,
                                                    DockerImage.pytorch_deepspeech]


    @classmethod
    def load(cls, overrides=[]):
        log.info("Loading Armory Environment")
        if len(overrides) != 0:
            overrides = {j[0]: j[1] for j in [i.split("=") for i in overrides]}
        args = {}
        for key in cls.__fields__.keys():
            args[key] = os.environ.get(key, None)
            if key in overrides:
                log.info(f"Overriding {key} with {overrides[key]}")
                args[key] = overrides[key]
        return cls(**args)

    def override(self, key:str, value):
        """Overide Environment Parameter

        key:

        """

    def override(self, vals: Union[dict, str]):
        """Override Evironment parameter
        vals:       Expects either space separated string of `key=value` pairs
                    e.g. 'credentials.s3_token=12345 paths.temporary_directory=/tmp
                    or `dict` with key in same format (e.g. {'paths.setup_file':'/tmp'})
        """
        if isinstance(vals, str):
            vals = {j[0]: j[1] for j in [i.split("=") for i in overrides]}
        elif isinstance(vals, dict):
            vals = vals
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
    else:
        raise Exception(
            "get_value must have one of `is_dir`, `is_bool`, `is_string` specified"
        )


def save_profile(profile_dict, filename):
    # TODO Refactor this to not write as ENV Vars
    # just go to .armory/profile.yml
    # Then search code for all refs to os.environ
    # and fix up
    if os.path.exists(filename):
        print(f"WARNING: Profile File: {filename} already exists...overwriting!!")
        response = ask_yes_no(None, "Are you sure?")
        if response in ("n"):
            print("Skipping Save!!")
            return
    print(f"Saving Profile to: {filename}")
    with open(filename, "w") as f:
        for k, v in profile_dict.items():
            line = f"export {k}={v}\n"
            f.write(line)


def setup_environment():
    main_dir = os.path.expanduser(os.path.join("~", ".armory"))
    profile = os.path.join(main_dir, "armory_profile")
    # TODO change to lower case
    DEFAULTS = dict(
        ARMORY_CONFIGURATION_DIRECTORY=(main_dir, "dir", None),
        ARMORY_GITHUB_TOKEN=(None, "str", None),
        ARMORY_EXECUTION_MODE=("native", "str", ["native", "docker"]),
        ARMORY_DATASET_DIRECTORY=(
            os.path.expanduser(os.path.join(main_dir, "datasets")),
            "dir",
            None,
        ),
        ARMORY_GIT_DIRECTORY=(
            os.path.expanduser(os.path.join(main_dir, "git")),
            "dir",
            None,
        ),
        ARMORY_OUTPUT_DIRECTORY=(
            os.path.expanduser(os.path.join(main_dir, "outputs")),
            "dir",
            None,
        ),
        ARMORY_SAVED_MODELS_DIRECTORY=(
            os.path.expanduser(os.path.join(main_dir, "saved_models")),
            "dir",
            None,
        ),
        ARMORY_TEMP_DIRECTORY=(
            os.path.expanduser(os.path.join(main_dir, "tmp")),
            "dir",
            None,
        ),
        ARMORY_VERIFY_SSL=(True, "bool", None),
    )

    while True:
        values = {}
        if os.path.isfile(profile):
            print("WARNING: this will overwrite existing armory profile.")
            print("    Press Ctrl-C to abort.")

        instructions = "\n".join(
            [
                "\nSetting up Armory Profile",
                f'    This profile will be stored at "{profile}"',
                "",
                "Please enter desired target directory for the following paths.",
                "    If left empty, the default path will be used.",
                "    Absolute paths (which include '~' user paths) are required.",
                "",
                "Stop at any time by pressing Ctrl-C.",
                "",
            ]
        )
        print(instructions)

        for k, v in DEFAULTS.items():
            values[k] = get_value(f"Set {k}", v[0], v[1], v[2])

        prompt = "\nConfiguration Items:\n"
        prompt += "\n".join([f"\t{k:35s}:\t\t{v}" for k, v in values.items()])
        response = ask_yes_no(prompt, "Save this Profile?")
        if response in ("y", ""):
            print(f"\nSaving Configuration to: {profile}")
            save_profile(values, profile)
            break
        else:
            print("Configuration Not Saved!!!")

    lines = "\n".join(
        [
            "",
            "Armory Setup Complete!!",
            "If not already done, you should add the following",
            "to your bash_profile:",
            "",
            "# Setting Armory Profile in Environment",
            f"ARMORY_PROFILE={profile}",
            "if [[ -f $ARMORY_PROFILE ]]; then",
            "   echo Setting Armory Environment",
            "   source $ARMORY_PROFILE",
            "fi",
            "\n" "Once complete, you will need to source your profile or",
            "start a new terminal session",
        ]
    )
    print(lines)
