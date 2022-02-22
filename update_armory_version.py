""" Update Armory Version

The armory version has to be specified in like 100 places, which is
crazy, but to make it sane for development, you can run this script
to update it appropriately
"""
import re
import glob
import json


def update_version(args):

    print(
        'Updating `armory.__init__.py` to use `__version__ = "{}"'.format(args.version)
    )
    with open("./armory/__init__.py", "r") as f:
        txt = f.read()

    txt = re.sub('__version__ = "(.*)"', '__version__ = "{}"'.format(args.version), txt)
    with open("./armory/__init__.py", "w") as f:
        f.write(txt)

    print("Updating All Scenario Configs")
    scenario_files = glob.glob("./scenario_configs/**/*.json", recursive=True)
    for fn in scenario_files:
        with open(fn, "r") as f:
            data = json.loads(f.read())

        if "docker_image" in data["sysconfig"]:
            if "twosixarmory" in data["sysconfig"]["docker_image"]:
                img = data["sysconfig"]["docker_image"].split(":")[0]
                data["sysconfig"]["docker_image"] = "{}:{}".format(img, args.version)
        print("Updating {} to: \n {}\n\n".format(fn, data))
        with open(fn, "w") as f:
            f.write(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="New Version to use")
    args = parser.parse_args()

    update_version(args)
