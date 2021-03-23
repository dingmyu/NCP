import functools
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function.

    See: https://arxiv.org/abs/1710.05941
    NOTE: Will consume much more GPU memory compared with inplaced ReLU.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'relu6': functools.partial(nn.ReLU6, inplace=True),
        'relu': functools.partial(nn.ReLU, inplace=True),
        'swish': Swish,
        'tanh': nn.Tanh,
    }[name]
    return active_fn


class MLP(nn.Module):
    def __init__(self,
                 activation='relu',
                 input_dim=30,
                 metrics=[3, 1],
                 shared_layer=256,
                 seperated_layer=128,
                 layer1_drop_ratio=0.5,
                 layer2_drop_ratio=0.5,
                 **kwargs):
        super(MLP, self).__init__()
        self.active_fn = get_active_fn(activation)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, shared_layer),
            nn.BatchNorm1d(shared_layer),
            self.active_fn(),
            nn.Dropout(p=layer1_drop_ratio),
        )
        nets = []
        for index, metric_num in enumerate(metrics):
            nets.append(
                nn.Sequential(
                    nn.Linear(shared_layer, seperated_layer),
                    nn.BatchNorm1d(seperated_layer),
                    self.active_fn(),
                    nn.Dropout(p=layer2_drop_ratio[index]),
                    nn.Linear(seperated_layer, metric_num)))

        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        x = self.fc(x)
        output = []
        for net in self.nets:
            output.append(net(x))
        return torch.cat(output, dim=1)
