import os
import pathlib
import unittest
import shutil

from armory import paths
from armory.utils.external_repo import download_and_extract_repo


class ExternalRepoTest(unittest.TestCase):
    def test_download(self):
        tmp_subdir = pathlib.Path(paths.host().tmp_dir, "test-external-repo-subdir")
        external_repo_dir = pathlib.Path(paths.get_external(tmp_subdir))
        repo = "twosixlabs/armory-example"
        repo_name = repo.split("/")[-1]

        download_and_extract_repo(repo, external_repo_dir=external_repo_dir)
        basedir = external_repo_dir / repo_name

        self.assertTrue(os.path.exists(basedir))
        self.assertTrue(os.path.isfile(basedir / "README.md"))
        shutil.rmtree(basedir)
        os.remove(external_repo_dir / (repo_name + ".tar.gz"))
        os.rmdir(external_repo_dir)
        os.rmdir(tmp_subdir)

    def test_download_branch(self):
        tmp_subdir = pathlib.Path(paths.host().tmp_dir, "test-external-repo-subdir")
        external_repo_dir = pathlib.Path(paths.get_external(tmp_subdir))
        repo = "twosixlabs/armory-example@armory-0.6"
        org_repo_name = repo.split("@")[0]
        repo_name = org_repo_name.split("/")[-1]

        download_and_extract_repo(repo, external_repo_dir=external_repo_dir)
        basedir = external_repo_dir / repo_name

        self.assertTrue(os.path.exists(basedir))
        self.assertTrue(os.path.isfile(basedir / "README.md"))
        shutil.rmtree(basedir)
        os.remove(external_repo_dir / (repo_name + ".tar.gz"))
        os.rmdir(external_repo_dir)
        os.rmdir(tmp_subdir)
