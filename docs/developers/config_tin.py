from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ArmoryCredentials:
    github_token: str
    s3_id: str
    s3_key: str


@dataclass
class ArmoryPaths:
    output_dir: Path
    local_git_dir: Path


def paths_by_mode(mode: str) -> ArmoryPaths:
    if mode == "docker":
        return ArmoryPaths(output_dir=Path("/output"), local_git_dir=Path("/git"))
    elif mode == "host":
        return ArmoryPaths(
            output_dir=Path("/tmp/armory"), local_git_dir=Path("/tmp/armory-local")
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


@dataclass
class ArmoryFlags:
    interactive: bool = False
    jupyter: bool = False
    host_port: Optional[str] = None
    check_run: bool = False
    num_eval_batches: Optional[int] = None
    skip_benign: bool = False
    skip_attack: bool = False
    skip_misclassified: bool = False
    validate_config: bool = False


@dataclass
class ConfigurationTin:
    mode: str
    flag: ArmoryFlags
    credential: ArmoryCredentials
    path: ArmoryPaths


# we know the control flags pretty early, so bundle them up after argparse
# if the class has defaults, you can construct with only fields you have
flag = ArmoryFlags(skip_attack=True)


# this constructor has no defaults so needs every field to be passed
tin = ConfigurationTin(
    mode="host",
    flag=flag,
    credential=ArmoryCredentials(github_token="ghp_", s3_id="AKIA", s3_key="2hG"),
    path=paths_by_mode("host"),
)

# access a field with a minimum of fuss and compile-time name checking
print(tin.credential.github_token)

# we're all consenting adults here, so modification is allowed, but
# this should be kicked out by code review
tin.credential.s3_id = None
