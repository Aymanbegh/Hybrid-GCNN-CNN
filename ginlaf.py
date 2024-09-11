

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
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
# from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.logging import init_wandb, log

from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

# import torch
# torch.manual_seed(0)
# import torch.nn.functional as F
# from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
# from torch_geometric.nn import GCNConv, GINConv
# from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.logging import init_wandb, log


# import os
# import torch
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)


from torch.nn import Parameter, Module, Sigmoid
import torch
import torch_scatter
import torch.nn.functional as F


hidden_channels=32
hidden_channels1=256
lr = 0.001
action='store_true'
epochs = 2

num_node_features = 2
num_edge_features = 1
num_classes = 2

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



class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()

        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)


class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps

    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)

        exps = (1. - e)/2. + x*e
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)
        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta
        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)

        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0
        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)
        res = num / den
        return res


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:
        #
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        print("Size edge data: {} ".format(len(edge_index)))
        print("edge data: {} ".format(edge_index.shape))
        out = self.propagate(edge_index, x=x, size=size)
        print("Size data: {} and {}".format(len(edge_index), len(x)))
        x_r = x[0]
        print("Size XR: {}".format(len(x_r)))
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=32, **kwargs):
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units

    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        print("TOPr X: {}".format(x.size()))
        print("TOPr X: {}".format(x))
        print("TOPr vale dim: {} and units {}".format(self.dim , self.units))
        # x = x.view((-1, self.dim * self.units * 2))
        x = x.view((self.dim * self.units * 2 * len(x)))
        print("TOPr : {}".format(len(x)))
        x = self.mlp(x)
        print("SUITE: {}".format(self.units))
        return x



class LAFNet(torch.nn.Module):
    def __init__(self,num_node_features, hidden_channels, num_classes):
        super(LAFNet, self).__init__()

        # num_node_features, hidden_channels, num_classes
        num_features = num_node_features
        # dim = hidden_channels
        dim = 32
        units = 3

        print("Size: {} and {} and {}".format(dim, num_features, units))
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINLAFConv(nn1, units=units, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINLAFConv(nn2, units=units, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINLAFConv(nn3, units=units, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINLAFConv(nn4, units=units, node_dim=dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINLAFConv(nn5, units=units, node_dim=dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

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