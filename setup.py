import codecs
import os
from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

tests_require = []
docs_require = []


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


with open("requirements.txt") as f:
    required_pkgs = f.read().splitlines()


setup(
    name="armory-testbed",
    version=get_version("armory/__init__.py"),
    description="Adversarial Robustness Test Bed",
    long_description=long_description,
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["armory = armory.__main__:main"]},
)
