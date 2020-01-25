"""
Script to format all JSON in git repo.

Format is the same as Python's built in json.tool
    However, it overcomes some command line errors in rewriting the same file

Usage: python -m tools.format_json [path] [--no-git]
    :argument path: Script will run from designated path instead of current working directory
    :argument --no-git: Whether to not to use git/gitignore to find files (default). Otherwise recursive.
"""

import argparse
import json
import os
import subprocess


def json_dumps_pretty(obj):
    return json.dumps(obj, sort_keys=True, indent=4) + "\n"


def json_tool(filepath) -> bool:
    """
    Equivalent to json.tool utility, except returns whether changes were made

    Returns whether changes were made 
    """

    with open(filepath) as f:
        content = f.read()

    obj = json.loads(content)

    output = json_dumps_pretty(obj)
    if output != content:
        with open(filepath, "w") as f:
            f.write(output)
        return True
    return False


def _inner_loop(filepaths):
    changed, stayed, errored = 0, 0, 0
    for filepath in filepaths:
        if filepath.lower().endswith(".json"):
            try:
                if json_tool(filepath):
                    print(f"reformatted {filepath}")
                    changed += 1
                else:
                    stayed += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"error: cannot format {filepath}: {e}")
                errored += 1

    def plural(count):
        if count == 1:
            return f"{count} file"
        return f"{count} files"

    changed, stayed, errored = [plural(x) for x in (changed, stayed, errored)]
    print(
        f"{changed} reformatted, {stayed} left unchanged, {errored} failed to reformat."
    )


def json_tool_recursive(rootdir=".", ignore_hidden=True):
    """
    Recursively combs root directory for json files, rewriting them.
    """
    if os.path.isfile(rootdir):
        filepaths = [rootdir]
    else:
        filepaths = []
        rootdir = rootdir or os.getcwd()
        for root, dirs, files in os.walk(rootdir):
            dirs.sort()
            files.sort()
            if ignore_hidden:
                dirs[:] = [x for x in dirs if not x.startswith(".")]
                files[:] = [x for x in files if not x.startswith(".")]
            for f in files:
                filepaths.append(os.path.join(root, f))

    _inner_loop(filepaths)


def json_tool_git(rootdir="."):
    """
    Uses git command line to find json files not ignored by git.

    If rootdir is ".", uses current working directory.
    """
    if not isinstance(rootdir, str):
        raise ValueError(f"rootdir must be a string, not {rootdir}")
    if not rootdir:
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
    _inner_loop(filepaths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lint JSON files.")
    parser.add_argument(
        "path", nargs="?", default=".", type=str, help="path to start looking from"
    )
    parser.add_argument(
        "--no-git",
        dest="no_git",
        nargs="?",
        const=True,
        default=False,
        type=bool,
        help="whether to use git to locate files (default) or os.walk",
    )
    parser.add_argument(
        "--hidden",
        nargs="?",
        const=True,
        default=False,
        type=bool,
        help="whether to consider hidden files and directories. Only applies to --no-git",
    )
    args = parser.parse_args()

    if args.no_git:
        json_tool_recursive(args.path, not args.hidden)
    else:
        json_tool_git(args.path)
