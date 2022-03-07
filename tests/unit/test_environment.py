import pytest
import armory.environment as ae
import os
import json


@pytest.fixture
def temp_armory_env_dict(tmp_path):
    armory_dir = str(tmp_path / ".armory")
    os.makedirs(armory_dir)
    default_paths = ae.Paths()
    default_paths.change_base(armory_dir)
    default_paths = default_paths.dict()
    for k, v in default_paths.items():
        os.makedirs(v)
    return (armory_dir, default_paths)


def test_default_paths():
    default = ae.Paths()
    for k in ae.Paths.__fields__.keys():
        assert os.path.dirname(getattr(default, k)) == str(ae.DEFAULT_ARMORY_DIRECTORY)


def test_path_contruction(temp_armory_env_dict):
    exp_dir, path_dict = temp_armory_env_dict
    paths = ae.Paths(**path_dict)
    for k, v in paths.dict().items():
        assert os.path.dirname(v) == exp_dir

    with pytest.raises(OSError):
        paths = ae.Paths(dataset_directory="/crap")
        paths.check()

    paths = ae.Paths(**path_dict)
    paths.change_base("new_dir")
    for k, v in paths.dict().items():
        assert os.path.dirname(v) == "new_dir"


def test_environment_construction(temp_armory_env_dict, tmp_path):
    armory_dir, path_pars = temp_armory_env_dict
    env = ae.EnvironmentParameters(
        execution_mode=ae.ExecutionMode.native,
        credentials=ae.Credentials(
            git_token="12345", s3_token="45678", verify_ssl=False
        ),
        paths=ae.Paths(**path_pars),
        images=[ae.DockerImage.base, ae.DockerImage.pytorch],
    )
    pars = env.dict()

    env2 = ae.EnvironmentParameters.parse_obj(pars)
    assert env == env2
    fname = os.path.join(armory_dir, "test_env.json")
    with open(fname, "w") as f:
        f.write(json.dumps(pars))

    env3 = ae.EnvironmentParameters.parse_file(fname)
    assert env == env3

    env4 = ae.EnvironmentParameters()
    assert env4.execution_mode == ae.ExecutionMode.native
    assert ae.DockerImage.tf2 in env4.images

    # Make sure it can json serialize
    env4.json()

    env4.execution_mode = "docker"
    print(env4.execution_mode)
    assert env4.execution_mode == ae.ExecutionMode.docker


def test_environment_load(temp_armory_env_dict, tmp_path):
    armory_dir, path_pars = temp_armory_env_dict
    env = ae.EnvironmentParameters(
        execution_mode=ae.ExecutionMode.native,
        credentials=ae.Credentials(
            git_token="12345", s3_token="45678", verify_ssl=False
        ),
        paths=ae.Paths(**path_pars),
        images=[ae.DockerImage.base, ae.DockerImage.pytorch],
    )
    env.profile = os.path.join(armory_dir, "profile")
    ae.save_profile(env.dict(), env.profile)
    env2 = ae.EnvironmentParameters.load(profile=env.profile)
    assert env == env2

    env3 = ae.EnvironmentParameters.load(
        profile=env.profile, overrides=["paths.dataset_directory=/tmp"]
    )
    assert env3.paths.dataset_directory == "/tmp"

    # Try with override that is not in environment
    env4 = ae.EnvironmentParameters.load(
        profile=env.profile, overrides=["blah.blah=foo"]
    )
    assert env4 == env
