import armory.paths as ap
import os

# these look at the paths selected to debug CI
# they are expected to fail but give clues about the runtime environment


def test_path_mode():
    assert ap.NO_DOCKER
    assert not ap.NO_DOCKER


def test_toplevel_path():
    assert ap.runtime_paths().cwd == "/workspace"
    assert ap.runtime_paths().cwd == os.getcwd()


def test_armory_dir():
    assert ap.runtime_paths().armory_dir == "/workspace/.armory"
