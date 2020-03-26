import threading


class ProgressPercentage(object):
    def __init__(self, client, bucket, filename, logger):
        self._filename = filename
        self._size = client.head_object(Bucket=bucket, Key=filename)["ContentLength"]
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._logger = logger

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            self._logger.info(
                f"{self._filename}  {self._seen_so_far} / {self._size}  ({percentage:.2f})"
            )
