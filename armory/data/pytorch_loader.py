import torch


class TFToTorchGenerator(torch.utils.data.IterableDataset):
    def __init__(self, tf_dataset):
        super().__init__()
        self.tf_dataset = tf_dataset

    def __iter__(self):
        for ex in self.tf_dataset.take(-1):
            x, y = ex
            # manually handle adverarial dataset
            if isinstance(x, tuple):
                x = (torch.from_numpy(x[0].numpy()), torch.from_numpy(x[1].numpy()))
                y = torch.from_numpy(y.numpy())
            # non-adversarial dataset
            else:
                x = torch.from_numpy(x.numpy())
                y = torch.from_numpy(y.numpy())
            yield x, y
