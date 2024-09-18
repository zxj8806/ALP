import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def softplus_absolute_value(Z):
    Z_product = torch.matmul(Z, Z.t())
    return torch.nn.functional.softplus(Z_product) - torch.nn.functional.softplus(-Z_product)


def dot_product_decode(Z):
    A_pred = torch.sigmoid(softplus_absolute_value(Z))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class ALP(nn.Module):
    def __init__(self, adj):
        super(ALP, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.bn = nn.BatchNorm1d(args.hidden2_dim, momentum=0.1)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)

        mean_norm = (self.mean - self.mean.mean(dim=0))

        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        mean_norm = gaussian_noise * torch.exp(self.logstd) * 0.01 + mean_norm

        return mean_norm

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred

