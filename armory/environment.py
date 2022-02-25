"""
Environment parameter names
"""

# TODO This is a relic...remove
ARMORY_VERSION = "ARMORY_VERSION"

from pydantic import BaseModel, validator
import os
from armory.logs import log


class EnvironmentConfiguration(BaseModel):
    """Armory Environment Configuration Context

    This Dataclass contains the environmental references
    necessary for armory execution.
    """

    ARMORY_CONFIGURATION_DIRECTORY: str = os.path.expanduser("~")
    ARMORY_GITHUB_TOKEN: str = None
    ARMORY_EXECUTION_MODE: str = "native"
    ARMORY_DATASET_DIRECTORY: str
    ARMORY_GIT_DIRECTORY: str
    ARMORY_OUTPUT_DIRECTORY: str
    ARMORY_SAVED_MODELS_DIRECTORY: str
    ARMORY_TEMP_DIRECTORY: str
    ARMORY_VERIFY_SSL: bool = True

    @validator("ARMORY_CONFIGURATION_DIRECTORY")
    def validate_armory_configuration_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid Configuration Directory: {dname}")
        return dname

    @validator("ARMORY_GITHUB_TOKEN")
    def validate_github_token(cls, token):
        # TODO: Figure out how to validate this
        return token

    @validator("ARMORY_EXECUTION_MODE")
    def validate_execution_mode(cls, mode):
        if mode.lower() not in ["native", "docker"]:
            raise ValueError(f"Invalid Execution Mode: {mode}")
        return mode.lower()

    @validator("ARMORY_DATASET_DIRECTORY")
    def validate_dataset_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid Dataset Directory: {dname}")
        return dname

    @validator("ARMORY_GIT_DIRECTORY")
    def validate_git_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid GIT Directory: {dname}")
        return dname

    @validator("ARMORY_OUTPUT_DIRECTORY")
    def validate_output_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid Output Directory: {dname}")
        return dname

    @validator("ARMORY_SAVED_MODELS_DIRECTORY")
    def validate_saved_models_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid Saved Models Directory: {dname}")
        return dname

    @validator("ARMORY_TEMP_DIRECTORY")
    def validate_temp_directory(cls, dname):
        dname = os.path.abspath(os.path.expanduser(dname))
        if not os.path.exists(dname):
            raise ValueError(f"Invalid TEMP Directory: {dname}")
        return dname

    @classmethod
    def load(cls, overrides=[]):
        log.info("Loading Armory Environment")
        if len(overrides) != 0:
            overrides = {j[0]:j[1] for j in [i.split("=") for i in overrides]}
        args = {}
        for key in cls.__fields__.keys():
            args[key] = os.environ.get(key,None)
            if key in overrides:
                log.info(f"Overriding {key} with {overrides[key]}")
                args[key] = overrides[key]
        return cls(**args)


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
