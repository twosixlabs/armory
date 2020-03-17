import os
import pathlib
import unittest
import shutil

from armory import paths
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        external_repo_dir = pathlib.Path(paths.host().external_repo_dir)
        repo = "twosixlabs/armory-example"
        repo_name = repo.split("/")[-1]

        download_and_extract_repo(repo, external_repo_dir=external_repo_dir)
        basedir = external_repo_dir / repo_name

        self.assertTrue(os.path.exists(basedir))
        self.assertTrue(os.path.isfile(basedir / "README.md"))
        shutil.rmtree(basedir)
        os.remove(external_repo_dir / (repo_name + ".tar.gz"))
        try:
            os.rmdir(external_repo_dir)
        except OSError:
            # Only delete if empty
            pass
        self.assertFalse(os.path.exists(external_repo_dir))
