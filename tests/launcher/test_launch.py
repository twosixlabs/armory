import subprocess
import pytest


def test_entrypoints(capfd):
    # check that entrypoints were installed correctl"y
    # TODO: look for a characteristic header string like "armory" use pytest capfd
    for entrypoint in "armory tiga engine".split():
        p = subprocess.run(entrypoint)
        assert p.returncode == 0, f"{entrypoint} failed to run"


@pytest.mark.docker_required
def test_docker_exec(tmp_path):
    from armory.launcher.launcher import execute_docker_cmd, DockerMount

    fn = tmp_path / "hello.txt"
    fn.write_text("Hello World")
    d = tmp_path / "subd"
    d.mkdir()

    mount = DockerMount(
        source=str(tmp_path), target="/my_space", type="bind", readonly=True
    )
    result = execute_docker_cmd("alpine", "ls /my_space", mounts=[mount])
    print(result)
    print("return code", result.returncode)
    assert result.returncode == 0, f"Docker Execution Failed: {result}"
    assert "hello.txt" in result.stdout.decode("utf-8")
