from dgl import nn as dglnn
import dgl
from dgl.nn.pytorch import GraphConv, GATConv
import torch
from torch import nn
from torch.nn import functional as F
import networkx as nx
import numpy as np

from ..utils.functions import get_activation


def build_dom_graph(dom):
    g = nx.DiGraph()
    element2idx = {element: idx for idx, element in enumerate(dom.iter())}
    for idx, element in enumerate(dom.iter()):
        g.add_node(idx)
    for idx, element in enumerate(dom.iter()):
        g.add_edge(idx, idx)
        parent = element.getparent()
        if parent is not None:
            g.add_edge(idx, element2idx[parent])
        children = element.getchildren()
        for child in children:
            g.add_edge(idx, element2idx[child])
        siblings = element.xpath('following-sibling::*[self::p]') + element.xpath(
            'preceding-sibling::*[self::p]')
        for sibling in siblings:
            g.add_edge(idx, element2idx[sibling])
    return g


def nx2dgl(g: nx.Graph):
    adj = np.array(nx.adjacency_matrix(g).todense())
    if len(np.where(~adj.any(axis=1))[0]):
        raise ValueError(g, "Graph has zero degree node.")
    graph_as_dgl = dgl.from_networkx(g)
    return dgl.add_self_loop(graph_as_dgl)


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size, num_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_size, activation=get_activation(activation)))
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=get_activation(activation)))
        self.layers.append(GraphConv(hidden_size, out_feats, activation=get_activation))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size, num_layers, num_heads, activation, dropout):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, hidden_size, num_heads))
        for i in range(num_layers - 2):
            self.layers.append(GATConv(hidden_size * num_heads, hidden_size, num_heads))
        self.layers.append(GATConv(hidden_size * num_heads, out_feats, 1))
        self.activation = get_activation(activation)

    def forward(self, g, h):
        for i, layer in enumerate(self.layers):
            h = layer(g, h).flatten(1)
        return h


class MPNN(nn.Module):
    def __init__(self, in_feats, out_feats,
                 num_rounds,
                 activation, dropout,
                 message_func, reduce_func):
        super(MPNN, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.num_rounds = num_rounds
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        self.message_func = message_func
        self.reduce_func = reduce_func

    def forward(self, g, h):
        g.ndata['h'] = h
        for i in range(self.num_rounds):
            g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('h')
        h = self.linear(h)
        return h
