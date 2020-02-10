import os
import unittest
import shutil

from armory import paths
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        repo = "twosixlabs/armory-example"
        repo_name = repo.split("/")[-1]

        download_and_extract_repo(repo)
        self.assertTrue(os.path.exists(f"{paths.EXTERNAL_REPOS}/{repo_name}"))
        self.assertTrue(os.path.isfile(f"{paths.EXTERNAL_REPOS}/{repo_name}/README.md"))

        shutil.rmtree(f"{paths.EXTERNAL_REPOS}/{repo_name}")
