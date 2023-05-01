import tensorflow as tf
import torch


class TFToTorchGenerator(torch.utils.data.IterableDataset):
    def __init__(self, tf_dataset):
        super().__init__()
        self.tf_dataset = tf_dataset

    def __iter__(self):
        for ex in self.tf_dataset.take(-1):
            x, y = ex
            # separately handle benign/adversarial data formats
            if isinstance(x, tuple):
                x_torch = (
                    torch.from_numpy(x[0].numpy()),
                    torch.from_numpy(x[1].numpy()),
                )
            else:
                x_torch = torch.from_numpy(x.numpy())

            # separately handle tensor/object detection label formats
            if isinstance(y, dict):
                y_torch = {}
                for k, v in y.items():
                    if isinstance(v, tf.Tensor):
                        y_torch[k] = torch.from_numpy(v.numpy())
                    else:
                        raise ValueError(
                            f"Expected all values to be of type tf.Tensor, but value at key {k} is of type {type(v)}"
                        )
            else:
                y_torch = torch.from_numpy(y.numpy())

            yield x_torch, y_torch


def get_pytorch_data_loader(ds):
    torch_ds = TFToTorchGenerator(ds)
    return torch.utils.data.DataLoader(
        torch_ds, batch_size=None, collate_fn=lambda x: x, num_workers=0
    )
