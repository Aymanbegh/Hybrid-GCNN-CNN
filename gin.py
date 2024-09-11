

import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import math
import csv
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os.path as osp

import numpy as np

import torch
torch.manual_seed(0)
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.logging import init_wandb, log
hidden_channels=1024
hidden_channels1=256
lr = 0.001
action='store_true'
epochs = 2

num_node_features = 2
num_edge_features = 1
num_classes = 2


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("used device: {}".format(device))

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, num_node_features, dim_h, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, num_classes)
        # self.batch = batch

    def forward(self, x, edge_index,batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
        # self.load_state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        # for key in list(state_dict.keys()):
        #     if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
        #         del state_dict[key]
        #
        #     # Also for backward compatibility with v1.0 weights, do this check
        #     if key.startswith('fpn.downsample_layers.'):
        #         if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
        #             del state_dict[key]
        # self.load_state_dict(state_dict)
        # self.state_dict


# model = GCN1(num_node_features, hidden_channels, num_classes)
# model = GCN(num_node_features, hidden_channels, num_classes)
# model_gcn = GIN(num_node_features, hidden_channels, num_classes)