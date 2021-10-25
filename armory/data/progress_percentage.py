import os
import sys
import threading

from tqdm import tqdm


class ProgressPercentage(tqdm):
    def __init__(self, client, bucket, filename, total):
        super().__init__(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=f"{filename} download",
            total=total,
            disable=False,
        )

    def __call__(self, bytes_amount):
        self.update(bytes_amount)


class ProgressPercentageUpload(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()
