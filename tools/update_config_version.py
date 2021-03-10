"""
Script to find and update all armory versions in JSON configs to current version

Usage: python -m tools.update_config_version [old version] [new_version]
"""

import argparse
import subprocess

STRINGS = [
    '"twosixarmory/pytorch:{}"',
    '"twosixarmory/pytorch-deepspeech:{}"',
    '"twosixarmory/tf1:{}"',
    '"twosixarmory/tf2:{}"',
]


def update_json_file(filepath, old_version, new_version):
    """
    Update json file, return True if file was updated
    """
    if not filepath.lower().endswith(".json"):
        raise ValueError("Not a json file")

    with open(filepath) as f:
        s = f.read()

    transcript = []
    for string in STRINGS:
        old = string.format(old_version)
        new = string.format(new_version)
        if old in s:
            s = s.replace(old, new)
            transcript.append((old, new))

    if transcript:
        print(f"Updated {filepath}:")
        for old, new in transcript:
            print(f"    {old} -> {new}")
        with open(filepath, "w") as f:
            f.write(s)
        print()
        return True
    return False


def git_json_files():
    """
    Uses git command line to find json files not ignored by git.
    """
    rootdir = "."

    def get_paths(cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            raise ValueError(f"Error calling {cmd}: {err}")
        out = str(out, encoding="utf-8")
        lines = out.splitlines()
        return lines

    filepaths = get_paths(["git", "ls-files", rootdir])
    filepaths.extend(
        get_paths(["git", "ls-files", rootdir, "--exclude-standard", "--others"])
    )
    filepaths = [x for x in filepaths if x.lower().endswith(".json")]
    return filepaths


def main(old_version, new_version):
    filepaths = git_json_files()
    print(f"Found {len(filepaths)} JSON files:")
    for f in filepaths:
        print(f"    {f}")
    print()

    for f in filepaths:
        update_json_file(f, old_version, new_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update JSON file config versions.")
    parser.add_argument("old_version", type=str, help="old version")
    parser.add_argument("new_version", type=str, help="new version")
    args = parser.parse_args()

    main(args.old_version, args.new_version)
