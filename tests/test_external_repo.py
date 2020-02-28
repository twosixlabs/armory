import os
import unittest
import shutil

from armory import paths
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        host_paths = paths.host()
        repo = "twosixlabs/armory-example"
        repo_name = repo.split("/")[-1]

        download_and_extract_repo(repo)
        self.assertTrue(os.path.exists(f"{host_paths.external_repo_dir}/{repo_name}"))
        self.assertTrue(
            os.path.isfile(f"{host_paths.external_repo_dir}/{repo_name}/README.md")
        )

        shutil.rmtree(f"{host_paths.external_repo_dir}/{repo_name}")
