"""
Utility functions for dealing with docker directories and mounted volumes
"""

import datetime
import logging
import os

from armory import paths

logger = logging.getLogger(__name__)


def tmp_output_subdir(retries=10):
    """
    Return (<subdir name>, <tmp subdir path>, <output subdir path>)

    retries - number of times to retry folder creation before returning an error
        if retries < 0, it will retry indefinitely.
        retries are necessary to prevent timestamp collisions.
    """
    tries = int(retries) + 1
    host_paths = paths.host()
    while tries:
        subdir = datetime.datetime.utcnow().isoformat()
        # ":" characters violate docker-py volume specifications
        subdir = subdir.replace(":", "-")
        # Use tmp_subdir for locking
        try:
            tmp_subdir = os.path.join(host_paths.tmp_dir, subdir)
            os.mkdir(tmp_subdir)
        except FileExistsError:
            tries -= 1
            if tries:
                logger.warning(f"Failed to create {tmp_subdir}. Retrying...")
            continue

        output_subdir = os.path.join(host_paths.output_dir, subdir)
        os.mkdir(output_subdir)
        return subdir, tmp_subdir, output_subdir

    raise ValueError("Failed to create tmp and output subdirectories")
