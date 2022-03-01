import pytest
import subprocess
import re
import os


def armory_version_tbi():
    """Expected Version of Armory to be installed in Docker Image"""
    expected_armory_version = subprocess.check_output(
        "python setup.py --version".split(" ")
    )
    expected_armory_version = expected_armory_version.decode("utf-8")
    expected_armory_version = expected_armory_version.replace("\n", "").strip()
    return expected_armory_version


def image_tag():
    av = armory_version_tbi().replace("+", "-")
    return av


def parse_output(output):
    if "--cache-from" in output:
        m = re.match(r"(.*?)--cache-from\s+(.*?)\s+(.*?)", output)
        cache = m.groups()[1]
    else:
        cache = None

    docker_file = re.match(r"(.*?)--file\s+(.*?)\s+(.*?)", output).groups()[1]
    docker_file = os.path.basename(docker_file)

    m = re.match(r"(.*?)--build-arg\s+base_image_tag=(.*?)\s+(.*?)", output)
    base_img_tag = None if m is None else m.groups()[1].replace("+", "-")

    m = re.match(r"(.*?)--build-arg\s+armory_version=(.*?)\s+(.*?)", output)
    armory_version = None if m is None else m.groups()[1]

    m = re.match(r"-t\s+twosixarmory/(.*?):(.*?)", output)
    tag = None if m is None else m.groups()[1].replace("+", "-")

    returned_output = {
        "cache": cache,
        "docker_file": docker_file,
        "base_img_tag": base_img_tag,
        "armory_version": armory_version,
        "image_tag": tag,
    }
    return returned_output


def get_cmd_output(cmd):
    output = subprocess.check_output(cmd.split(" "))
    output = output.decode("utf-8")
    output = str(output).split("->")[1]
    output = output.replace("\n", "")
    return output


@pytest.mark.parametrize(
    "cmd, expected_outputs",
    [
        (
            "bash docker/build.sh -f base --dry-run",
            {
                "cache": f"twosixarmory/base:{image_tag()}",
                "docker_file": "Dockerfile-base",
                "base_img_tag": None,
                "armory_version": None,
                "image_tag": f"twosixarmory/base:{image_tag()}",
            },
        ),
        (
            "bash docker/build.sh -f pytorch --dry-run",
            {
                "cache": f"twosixarmory/pytorch:{image_tag()}",
                "docker_file": "Dockerfile-pytorch",
                "base_img_tag": image_tag(),
                "armory_version": armory_version_tbi(),
                "image_tag": f"twosixarmory/pytorch:{image_tag()}",
            },
        ),
        (
            "bash docker/build.sh -f pytorch --dry-run -av 0.12.12",
            {
                "cache": f"twosixarmory/pytorch:{image_tag()}",
                "docker_file": "Dockerfile-pytorch",
                "base_img_tag": image_tag(),
                "armory_version": "0.12.12",
                "image_tag": f"twosixarmory/pytorch:{image_tag()}",
            },
        ),
    ],
)
def test_docker_build_script(cmd, expected_outputs):
    output = get_cmd_output(cmd)
    vals = parse_output(output)
    for name in expected_outputs:
        try:
            assert vals[name] == expected_outputs[name]
        except AssertionError as e:
            print(f"Output: {name} did not match")
            raise e
