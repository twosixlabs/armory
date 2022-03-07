from armory.utils import rgetattr, rsetattr, rhasattr
from armory.environment import EnvironmentParameters


def test_dot_notation():

    env = EnvironmentParameters()
    pth = env.paths.dataset_directory

    pth2 = rgetattr(env, "paths.dataset_directory")
    assert pth == pth2

    pth3 = "blah/blah"
    rsetattr(env, "paths.dataset_directory", pth3)
    assert env.paths.dataset_directory == pth3

    assert rhasattr(env, "paths.temporary_directory")
