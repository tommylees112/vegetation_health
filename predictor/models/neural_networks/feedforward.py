from torch import nn

from pathlib import Path

from .nn_base import NNBase
from ...preprocessing import VALUE_COLS, VEGETATION_LABELS


class FeedForward(NNBase):
    """A simple feedforward neural network
    """

    def __init__(self, arrays=Path('data/processed/arrays'), hide_vegetation=False):

        features_per_month = len(VALUE_COLS)
        if hide_vegetation:
            features_per_month -= len(VEGETATION_LABELS)

        num_features = features_per_month * 11

        super().__init__(LinearModel(num_features, [num_features, 10 * num_features], 0.25),
                         arrays, hide_vegetation)


class LinearModel(nn.Module):

    def __init__(self, input_size, layer_sizes, dropout):
        super().__init__()
        layer_sizes.insert(0, input_size)

        self.dense_layers = nn.ModuleList([
            LinearBlock(in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i], dropout=dropout) for
            i in range(1, len(layer_sizes))
        ])

        self.final_dense = nn.Linear(in_features=layer_sizes[-1], out_features=1)

        self.init_weights()

    def init_weights(self):
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.linear.weight.data)
            # http://cs231n.github.io/neural-networks-2/#init
            # see: Initializing the biases
            nn.init.constant_(dense_layer.linear.bias.data, 0)

        nn.init.kaiming_uniform_(self.final_dense.weight.data)
        nn.init.constant_(self.final_dense.bias.data, 0)

    def forward(self, x):
        # flatten
        x = x.view(x.shape[0], -1)
        for layer in self.dense_layers:
            x = layer(x)

        return self.final_dense(x)


class LinearBlock(nn.Module):
    """
    A linear layer followed by batchnorm, a ReLU activation, and dropout
    """

    def __init__(self, in_features, out_features, dropout=0.25):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear(x))
        return self.dropout(x)
