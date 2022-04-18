import subprocess


def test_entrypoints(capfd):
    # check that entrypoints were installed correctl"y
    # TODO: look for a characteristic header string like "armory" use pytest capfd
    for entrypoint in "armory tiga engine".split():
        p = subprocess.run(entrypoint)
        assert p.returncode == 0, f"{entrypoint} failed to run"
