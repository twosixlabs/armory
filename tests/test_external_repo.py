import os
import json
import unittest
import shutil
from armory.utils.external_repo import download_and_extract


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        with open('tests/test_data/external_repo.json') as f:
            config = json.load(f)

        repo_name = config["external_github_repo"].split('/')[-1]

        download_and_extract(config)
        self.assertTrue(os.path.exists(f"external_repos/{repo_name}"))

        shutil.rmtree("external_repos")