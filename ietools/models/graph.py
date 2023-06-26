from collections import deque
from math import sqrt

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


def vis_distance(node1, node2):
    # calculate Euclidean distance between two nodes
    x1, y1 = node1.attrib['x'], node1.attrib['y']
    x2, y2 = node2.attrib['x'], node2.attrib['y']
    return sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)


def depth_distance(node1, node2):
    # calculate depth distance between two nodes in the DOM tree
    depth1 = len(node1.xpath('ancestor::*'))
    depth2 = len(node2.xpath('ancestor::*'))
    return abs(depth2 - depth1)


def hop_distance(node1, node2):
    # calculate hop distance between two nodes in the DOM tree
    queue = deque([(node1, 0)])
    visited = set()
    while queue:
        current_node, current_depth = queue.popleft()
        if current_node == node2:
            return current_depth
        visited.add(current_node)
        for child in current_node.iterchildren():
            if child not in visited:
                queue.append((child, current_depth + 1))
    return float('inf')


def find_nearest_neighbors(dom_nodes, nodeset, distance):
    result = []
    for node in dom_nodes:
        min_distance = float('inf')
        nearest_node = None
        for neighbor in nodeset:
            d = distance(node, neighbor)
            if d < min_distance:
                min_distance = d
                nearest_node = neighbor
        result.append((nearest_node, min_distance))
    return result


def build_text_node_graph(dom, distance):
    g = nx.DiGraph()
    text_nodes = [element for element in dom.iter() if element.text.strip() != ""]
    text2idx = {element: idx for idx, element in enumerate(dom.iter()) if element.text.strip() != ""}
    for element, idx in text2idx.items():
        g.add_node(idx)
    nearest = find_nearest_neighbors(text_nodes, text_nodes, distance)
    for element, idx in text2idx.items():
        g.add_edge(idx, idx)
        g.add_edge(idx,
                   text2idx[nearest[idx][0]],
                   weight=nearest[idx][1])
        g.add_edge(text2idx[nearest[idx][0]],
                   idx,
                   weight=nearest[idx][1])
    return g


def nx2dgl(g: nx.Graph):
    adj = np.array(nx.adjacency_matrix(g).todense())
    if len(np.where(~adj.any(axis=1))[0]):
        raise ValueError(g, "Graph has zero degree node.")
    graph_as_dgl = dgl.from_networkx(g)
    return graph_as_dgl


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size, num_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, hidden_size, activation=get_activation(activation)))
        for i in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=get_activation(activation)))
        self.layers.append(GraphConv(hidden_size, out_feats, activation=get_activation(activation)))

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
