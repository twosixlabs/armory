from pathlib import Path
import os
import armory.paths as ap

# these look at the paths selected to debug CI
# they are expected to fail but give clues about the runtime environment


def test_starting_config():
    # these all pass in a container at startup
    assert ap.NO_DOCKER
    assert ap.runtime_paths().cwd == "/armory-repo"
    assert ap.runtime_paths().cwd == os.getcwd()
    assert ap.runtime_paths().armory_dir == "/root/.armory"


def test_config_present():
    armory = Path(ap.runtime_paths().armory_dir)
    assert armory.exists()
    assert armory.is_dir()
    config = armory / "config.json"
    assert config.exists()
    body = config.read_text()
    assert body.startswith("{")
    assert body.endswith("}")
    assert "tmp_dir" in body
