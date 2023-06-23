from dgl import nn as dglnn
import dgl
import torch
from torch import nn
import networkx as nx
import numpy as np

from ..utils.functions import get_activation


def build_dom_graph(dom):
    g = nx.DiGraph()
    element2idx = {element: idx for idx, element in enumerate(dom.iter())}
    for idx, element in enumerate(dom.iter()):
        g.add_node(idx)
    for idx, element in enumerate(dom.iter()):
        parent = element.getparent()
        if parent is not None:
            g.add_edge(idx, element2idx[parent])
        children = element.getchildren()
        for child in children:
            g.add_edge(idx, element2idx[child])
    return g


def nx2dgl(g: nx.Graph):
    adj = np.array(nx.adjacency_matrix(g).todense())
    if len(np.where(~adj.any(axis=1))[0]):
        raise ValueError(g, "Graph has zero degree node.")
    graph_as_dgl = dgl.from_networkx(g)
    return dgl.add_self_loop(graph_as_dgl)


class GNN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.5):
        super(GNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.activation = get_activation(activation)
        self.dropout = dropout

        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.message_func = lambda edges: {'msg': edges.src['h']}
        self.reduce_func = lambda nodes: {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

    def forward(self, g, x):
        h = self.activation(self.linear(x))
        h = self.dropout_layer(h)
        g.ndata['h'] = h
        g.update_all(message_func=self.message_func,
                     reduce_func=self.reduce_func)
        h = g.ndata.pop('h')
        return h



