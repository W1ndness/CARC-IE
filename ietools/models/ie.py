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


def fetch_text_nodes_xpath_embeddings(xpath_embeddings, ids):
    text_node_embeddings = []
    for idx, indices in enumerate(ids):
        xpath_embedding = xpath_embeddings[idx]
        text_node_embedding = torch.index_select(xpath_embedding, 0, indices)
        text_node_embeddings.append(text_node_embedding)
    return torch.stack(text_node_embeddings)



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

    def forward(self, batch):
        ids = batch['ids']
        text_embeddings = batch['text_embeddings']
        xpath_tags_seq = batch['xpath_tags_seq']
        xpath_subs_seq = batch['xpath_subs_seq']
        graphs = batch['graphs']
        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)
        h = self.gnn(graphs, xpath_embeddings)
        text_node_xpath_embeddings = fetch_text_nodes_xpath_embeddings(h, ids)
        text_node_embeddings = torch.cat([text_embeddings, text_node_xpath_embeddings], dim=1)
        output = self.classifier(text_node_embeddings)
        return output




