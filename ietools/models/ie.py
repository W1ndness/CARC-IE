from typing import List

import torch
from torch import nn
from transformers import MarkupLMConfig
from torch.nn import functional as F

from classifier import MLPClassifier
from encoder import XPathEmbeddings
from graph import GNN
from ..utils.functions import get_activation


class ModelConfig:
    def __init__(self,
                 text_embed_size: int,
                 gnn_out_size: int,
                 xpath_embeddings_config: MarkupLMConfig,
                 gnn_activation: str,
                 gnn_aggregator,
                 gnn_dropout: float,
                 mlp_hidden_dims: List[int],
                 mlp_activation: str,
                 mlp_dropout: float,
                 mlp_batch_norm: bool,
                 num_classes: int):
        self.text_embed_size = text_embed_size
        self.gnn_out_size = gnn_out_size
        self.xpath_embeddings_config = xpath_embeddings_config
        self.gnn_activation = gnn_activation
        self.gnn_aggregator = gnn_aggregator
        self.gnn_dropout = gnn_dropout
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout
        self.mlp_batch_norm = mlp_batch_norm
        self.num_classes = num_classes


class Model(nn.Module):
    def __init__(self,
                 config: ModelConfig):
        super(Model, self).__init__()
        self.xpath_embeddings = XPathEmbeddings(config.xpath_embeddings_config)
        self.gnn = GNN(config.text_embed_size + config.xpath_embeddings_config.hidden_size,
                       config.gnn_out_size,
                       get_activation(config.gnn_activation),
                       config.gnn_aggregator)
        self.classifier = MLPClassifier(config.gnn_out_size,
                                        config.num_classes,
                                        config.mlp_hidden_dims,
                                        get_activation(config.mlp_activation),
                                        config.mlp_dropout,
                                        config.mlp_batch_norm)

    def forward(self,
                text_embeddings,
                xpath_tags_seq,
                xpath_subs_seq,
                g):
        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)
        node_embeddings = torch.concat([F.normalize(text_embeddings),
                                        F.normalize(xpath_embeddings)], dim=1)
        h = self.gnn(g, node_embeddings)
        output = self.classifier(h)
        return output




