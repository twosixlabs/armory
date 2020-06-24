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
