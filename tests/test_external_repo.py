import os
import json
import unittest
import shutil
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        # Skip test if token is unavailable
        if os.environ.get("GITHUB_TOKEN") is None:
            return

        with open("tests/test_data/external_repo.json") as f:
            config = json.load(f)

        repo_name = config["external_github_repo"].split("/")[-1]

        download_and_extract_repo(config["external_github_repo"])
        self.assertTrue(os.path.exists(f"external_repos/{repo_name}"))
        self.assertTrue(os.path.isfile(f"external_repos/{repo_name}/README.md"))

        shutil.rmtree("external_repos")
