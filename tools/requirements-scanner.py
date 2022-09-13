#! /usr/bin/env python3
"""
Scans version pinned python requirements for updates.
"""


import sys
import pip
import json
import argparse
import subprocess
import pkg_resources

from pathlib import Path


class RequirementsScanner:
    """
    Example:
        >>> scanner = RequirementsScanner('.')
        >>> scanner.scan()
        >>> print(scanner)
    """
    VERBOSE   = False
    ERROR_MSG = "ERROR"

    def __init__(self, path, verbose=False):
        self.path = Path(path).resolve()
        self.VERBOSE   = verbose
        self.installed = {pkg.key for pkg in pkg_resources.working_set}
        self.requirements = {}



    def __repr__(self):
        requirements_dict = {f.name: content for (f, content) in self.requirements.items()}
        return json.dumps(requirements_dict, indent=2)


    def scan(self):
        files = [f for f in self.path.iterdir() if f.is_file() and f.name.endswith('.txt')]
        requirements = { }
        for f in files:
            print(f"Parsing {f.name}...")
            content = self.parse(f.read_text())
            # Check pinned requirements.
            if any([version['pinned'] for package, version in content.items()]):
                for package, version in content.items():
                    if version['pinned'] != None:
                        content[package]['latest'] = self.latest(package)
            requirements[f] = content
        self.requirements = requirements
        return self.requirements


    def parse(self, text):
        data = text.splitlines()
        packages    = { }
        split_chars = ['==', '>=', '<=', '>', '<', '!=']
        skip_chars  = ['-', '#', 'git+', 'hg+', 'svn+', 'bzr+', 'cvs+', 'http+', 'https+', 'file+']
        for line in data:
            line = line.strip()
            # Skip comments and empty lines
            if line == '' or any(map(line.startswith, skip_chars)):
                continue
            # Remove any inline comments.
            package, version = line.split('#')[0].strip(), None
            # Check if the package has a version.
            version_pinned = [split_char in package for split_char in split_chars]
            if any(version_pinned):
                deliminator = split_chars[version_pinned.index(True)]
                package, version = package.split(deliminator)
            packages[package] = {
                'pinned': version,
                'latest': None,
            }
        return packages


    def latest(self, package):
        # self.pip = pip.main if hasattr(pip, 'main') else pip._internal.main
        # self.pip(['index', 'versions', package])
        latest = self.ERROR_MSG
        try:
            versions = subprocess.check_output(['pip', 'index', 'versions', package])
            versions = versions.decode('utf-8', errors='replace').lower()
            versions = versions.split('versions:')[-1].split(', ')
            latest   = versions[0].strip()
        except subprocess.CalledProcessError:
            latest = self.ERROR_MSG
        return latest


    def update(self):
        requirements = self.requirements
        errors = []
        for f, packages in requirements.items():
            file_lines = f.read_text().splitlines()
            needs_updating = [package for package, version in packages.items() if version['pinned'] != version['latest']]
            if len(needs_updating):
                for i, line in enumerate(file_lines):
                    for package in needs_updating:
                        if line.startswith(package):
                            pinned_version = packages[package]['pinned']
                            latest_version = packages[package]['latest']
                            if latest_version == self.ERROR_MSG:
                                errors.append(f"{package} in {f.name} not found on PyPI.")
                            else:
                                print(f"Updating {package} in {f.name} to version {latest_version}...")
                                file_lines[i] = line.replace(pinned_version, latest_version)
                f.write_text("\n".join(file_lines))
        if len(errors) > 0:
            print(f"\n{'='*42}")
            print("\n\t - ERROR: ".join(errors))
            print(f"{'='*42}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--repo", help="Path to the repo directory", type=Path)

    args = parser.parse_args()

    if not args.repo or not Path(args.repo).resolve().exists():
        print("Please provide a path to the repo directory.")
        exit(1)

    scanner = RequirementsScanner(args.repo, verbose=True)
    # Scan the repo.
    scanner.scan()

    # Optional, update the requirements.
    # scanner.update()

    # print(repr(scanner))
