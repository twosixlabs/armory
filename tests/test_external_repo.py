import os
import unittest
import shutil
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        # Skip test if token is unavailable
        if os.environ.get("GITHUB_TOKEN") is None:
            return

        repo = "twosixlabs/armory-performer"
        repo_name = repo.split("/")[-1]

        download_and_extract_repo(repo)
        self.assertTrue(os.path.exists(f"external_repos/{repo_name}"))
        self.assertTrue(os.path.isfile(f"external_repos/{repo_name}/README.md"))

        shutil.rmtree("external_repos")
