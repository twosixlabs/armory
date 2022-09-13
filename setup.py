import os
import codecs
import datetime

from pathlib import Path

from setuptools import setup
from setuptools import find_packages


PACKAGE_NAME     = "armory-testbed"
LONG_DESCRIPTION = Path("README.md").read_text()

required_pkgs = Path("requirements.txt").read_text().splitlines()
tests_require = Path("test-requirements.txt").read_text().splitlines()
docs_require  = [ ]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# Allow installation without git repository, e.g. inside Docker.
if os.path.exists('.git'):
    kwargs = dict(
        setup_requires = ["setuptools_scm"],
        use_scm_version = {
            "root": ".",
            "relative_to": __file__,
            "local_scheme": "node-and-timestamp",
        }
    )
else:
    app_version = os.getenv('armory_version') if bool(os.getenv('armory_version')) else '0+d'+datetime.date.today().strftime('%Y%m%d')
    kwargs = dict(version = app_version)


setup(
    name=PACKAGE_NAME,
    description="Adversarial Robustness Test Bed",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Two Six Labs",
    author_email="armory@twosixlabs.com",
    url="https://github.com/twosixlabs/armory",
    license="MIT",
    install_requires=required_pkgs,
    tests_require=tests_require,
    extras_require={"tests": tests_require, "docs": docs_require},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["armory = armory.__main__:main"]},
    **kwargs
)
