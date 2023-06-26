from typing import List

import dgl
import torch
from torch import nn
from transformers import MarkupLMConfig
from torch.nn import functional as F
from d2l import torch as d2l

from .classifier import MLPClassifier
from .encoder import XPathEmbeddings
from .graph import MPNN, GCN, GAT

devices = d2l.try_all_gpus()


class ModelConfig:
    def __init__(self,
                 text_embed_size: int,
                 gnn_model: str,
                 gnn_out_size: int,
                 xpath_embeddings_config: MarkupLMConfig,
                 gnn_hidden_size: int,
                 gnn_num_layers: int,
                 gnn_activation: str,
                 mlp_hidden_dims: List[int],
                 mlp_activation: str,
                 mlp_dropout: float,
                 mlp_batch_norm: bool,
                 num_classes: int,
                 gnn_dropout: float = None,
                 gat_num_heads: int = None,
                 mpnn_msg_func=None,
                 mpnn_update_func=None):
        self.text_embed_size = text_embed_size
        if gnn_model not in ['gcn', 'gat', 'mpnn']:
            raise ValueError('Invalid gnn_model: {}'.format(gnn_model))
        self.gnn_model = gnn_model
        self.gnn_out_size = gnn_out_size
        self.xpath_embeddings_config = xpath_embeddings_config
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_num_layers = gnn_num_layers
        self.gnn_activation = gnn_activation
        self.gnn_dropout = gnn_dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batch_norm = mlp_batch_norm
        self.num_classes = num_classes
        if self.gnn_model == 'gat' and gat_num_heads is None:
            raise ValueError('gat_num_headers must be specified for gat model')
        self.gat_num_heads = gat_num_heads
        if self.gnn_model == 'mpnn' and (mpnn_msg_func is None or mpnn_update_func is None):
            raise ValueError('mpnn_msg_func and mpnn_update_func must be specified for mpnn model')
        self.mpnn_msg_func = mpnn_msg_func
        self.mpnn_update_func = mpnn_update_func


def fetch_text_nodes_xpath_embeddings(xpath_embeddings, ids):
    text_node_embeddings = []
    for idx, indices in enumerate(ids):
        xpath_embedding = xpath_embeddings[idx]
        indices = indices.to(devices[0])
        text_node_embedding = torch.index_select(xpath_embedding, 0, indices)
        text_node_embeddings.append(text_node_embedding)
    return torch.cat(text_node_embeddings, dim=0)


class Model(nn.Module):
    def __init__(self,
                 config: ModelConfig):
        super(Model, self).__init__()
        self.xpath_embeddings = XPathEmbeddings(config.xpath_embeddings_config)
        if config.gnn_model == 'mpnn':
            self.gnn = MPNN(config.xpath_embeddings_config.hidden_size,
                            config.gnn_out_size,
                            config.gnn_num_layers,
                            config.gnn_activation,
                            config.gnn_dropout,
                            config.mpnn_msg_func,
                            config.mpnn_update_func)
        elif config.gnn_model == 'gcn':
            self.gnn = GCN(config.xpath_embeddings_config.hidden_size,
                           config.gnn_out_size,
                           config.gnn_hidden_size,
                           config.gnn_num_layers,
                           config.gnn_activation,
                           config.gnn_dropout)
        elif config.gnn_model == 'gat':
            self.gnn = GAT(config.xpath_embeddings_config.hidden_size,
                           config.gnn_out_size,
                           config.gnn_hidden_size,
                           config.gnn_num_layers,
                           config.gat_num_heads,
                           config.gnn_activation,
                           config.gnn_dropout)
        self.classifier = MLPClassifier(config.text_embed_size + config.gnn_out_size,
                                        config.num_classes,
                                        config.mlp_hidden_dims,
                                        config.mlp_activation,
                                        config.mlp_dropout,
                                        config.mlp_batch_norm)

    def forward(self, batch):
        ids = batch['ids']
        text_embeddings = batch['text_embeddings']
        xpath_tags_seq = batch['xpath_tags_seq']
        xpath_subs_seq = batch['xpath_subs_seq']
        graphs = batch['graphs']
        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)
        xpath_embeddings = torch.squeeze(xpath_embeddings)
        h = self.gnn(graphs, xpath_embeddings)
        h = torch.split(h, list(graphs.batch_num_nodes()), dim=0)
        text_node_xpath_embeddings = fetch_text_nodes_xpath_embeddings(h, ids)
        assert text_node_xpath_embeddings.size(1) == text_embeddings.size(1)
        text_node_embeddings = torch.cat([F.normalize(text_embeddings), F.normalize(text_node_xpath_embeddings)],
                                         dim=1)
        output = self.classifier(text_node_embeddings)
        return output
