import torch
from torch import nn
from ..utils.functions import get_activation


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_dims,
                 activation,
                 dropout=0.0,
                 batch_norm=False):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = get_activation(activation)
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Linear(last_dim, hidden_dim)
            self.layers.append(layer)
            if batch_norm:
                bn = nn.BatchNorm1d(hidden_dim)
                self.layers.append(bn)
            activation = get_activation(activation)
            self.layers.append(activation)
            if dropout:
                dp = nn.Dropout(p=dropout)
                self.layers.append(dp)
            last_dim = hidden_dim
        # 添加输出层
        self.output_layer = nn.Linear(last_dim, output_dim)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X = self.output_layer(X)
        return X