import os
import re
import subprocess

import pytest

pytestmark = [pytest.mark.docker_required, pytest.mark.unit]


def get_cmd_output(cmd):
    output = subprocess.check_output(cmd.split(" "))
    output = output.decode("utf-8")
    output = str(output).split("->")[1]
    output = output.replace("\n", "").strip()
    return output


@pytest.fixture
def armory_version_tbi():
    """Expected Version of Armory to be installed in Docker Image"""
    try:
        expected_armory_version = subprocess.check_output(
            "python setup.py --version".split(" ")
        )
    except subprocess.CalledProcessError:
        print("armory .git not avaiable...trying armory")
        expected_armory_version = subprocess.check_output("armory version".split(" "))
    expected_armory_version = expected_armory_version.decode("utf-8")
    expected_armory_version = expected_armory_version.replace("\n", "").strip()
    return expected_armory_version


def get_short(value):
    return ".".join(value.split(".")[:4])


@pytest.fixture
def image_tag(armory_version_tbi):
    av = armory_version_tbi.replace("+", "-")
    tag = get_short(av)
    return tag


@pytest.mark.parametrize(
    "img, opt",
    [
        #        ("base", ""),
        ("pytorch", ""),
        ("tf2", ""),
        ("pytorch-deepspeech", ""),
        ("pytorch-deepspeech", "--no-cache"),
        #        ("base", "--no-cache"),
    ],
)
def test_build_script(img, opt, image_tag, armory_version_tbi):
    cmd = f"python docker/build.py {img} --dry-run {opt}"
    output = get_cmd_output(cmd)
    assert output.startswith("docker build")
    assert "--force-rm" in output

    if "--no-cache" in opt:
        assert "--no-cache" in output

    docker_file = re.match(r"(.*?)--file\s+(.*?)\s+(.*?)", output).groups()[1]
    docker_file = os.path.basename(docker_file)

    assert docker_file == f"Dockerfile-{img}"

    if img != "base":
        base_img_tag = re.match(
            r"(.*?)--build-arg\s+base_image_tag=(.*?)\s+(.*?)", output
        ).groups()[1]
        assert base_img_tag == image_tag

        armory_version = re.match(
            r"(.*?)--build-arg\s+armory_version=(.*?)\s+(.*?)", output
        ).groups()[1]
        assert get_short(armory_version) == get_short(armory_version_tbi)

    image = re.match(r"(.*?)\s+-t\s+twosixarmory/(.*?)\s+", output).groups()[1]
    assert image == f"{img}:{image_tag}"
