

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
# hidden_channels1=256
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

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
