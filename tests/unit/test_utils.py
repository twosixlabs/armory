import pytest
from armory.utils import rgetattr, rsetattr, rhasattr, parse_overrides
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


@pytest.mark.parametrize(
    "value, error, exp_len, exp_output",
    [
        ("a.b.c=4", None, 1, [("a.b.c", "4")]),
        ("a:4", ValueError, None, None),
        ("a.b.c=4 c.d=7", None, 2, [("a.b.c", "4"), ("c.d", "7")]),
        (["a.b.c=4", "c.d=8"], None, 2, [("a.b.c", "4"), ("c.d", "8")]),
    ],
)
def test_parse_overrides(value, error, exp_len, exp_output):
    if error:
        with pytest.raises(error):
            parse_overrides(value)
    else:
        output = parse_overrides(value)
        assert len(output) == exp_len
        assert output == exp_output
