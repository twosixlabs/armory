from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn

nclasses = 43  # GTSRB has 43 classes


class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, input):
        return input.permute(0, 3, 1, 2)


def Net():
    conv1 = nn.Conv2d(3, 1, kernel_size=1)
    conv2 = nn.Conv2d(1, 29, kernel_size=5)
    maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    conv3 = nn.Conv2d(29, 59, kernel_size=3)
    maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    conv4 = nn.Conv2d(59, 74, kernel_size=3)
    maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    fc1 = nn.Linear(1184, 300)
    fc2 = nn.Linear(300, nclasses)
    conv1_bn = nn.BatchNorm2d(1)
    conv2_bn = nn.BatchNorm2d(29)
    conv3_bn = nn.BatchNorm2d(59)
    conv4_bn = nn.BatchNorm2d(74)
    dense1_bn = nn.BatchNorm1d(300)
    ReLU = nn.ReLU()

    return nn.Sequential(
        Permute(),
        conv1,
        conv1_bn,
        ReLU,
        conv2,
        conv2_bn,
        ReLU,
        maxpool2,
        conv3,
        conv3_bn,
        ReLU,
        maxpool3,
        conv4,
        conv4_bn,
        ReLU,
        maxpool4,
        nn.Flatten(),
        fc1,
        ReLU,
        dense1_bn,
        fc2,
        nn.LogSoftmax(),
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_art_model_wrapper(model, model_kwargs, wrapper_kwargs, weights_path=None):
    model.to(DEVICE)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)

    loss = torch.nn.CrossEntropyLoss()
    wrapped_model = PyTorchClassifier(
        model,
        loss=loss,
        optimizer=torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True
        ),
        input_shape=(3, 48, 48),
        clip_values=(0.0, 1.0),
        nb_classes=nclasses,
        **wrapper_kwargs
    )
    return wrapped_model


def get_art_model(model_kwargs, wrapper_kwargs, weights_path=None):
    model = Net()
    return get_art_model_wrapper(model, model_kwargs, wrapper_kwargs, weights_path)
