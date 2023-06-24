from itertools import chain
from typing import List

import torch
from absl import logging
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import dgl

from ..utils.functions import get_dom_tree
from .graph import build_dom_graph, nx2dgl
from ..models.encoder import TextEncoder
from ..utils import swde as constants

encoder = TextEncoder()


def get_texts(dom, text_nodes_ids):
    return [element.text.strip() for idx, element in enumerate(dom.iter()) if idx in text_nodes_ids]


def mapping_labels(labels, label_map):
    return [label_map[label] for label in labels]


class Sample:
    def __init__(self, page_info: dict, page_html, website, label_map):
        self.idx = page_info["page_id"]
        nodes_info = page_info["nodes_info"]
        self.text_nodes_ids = [node["idx"] for node in nodes_info]
        dom = get_dom_tree(page_html, website)
        self.text_nodes_texts = get_texts(dom, self.text_nodes_ids)
        if not self.text_nodes_texts:
            self.text_nodes_texts = [""]
        self.text_nodes_labels = mapping_labels([node["label"] for node in nodes_info], label_map)
        self.xpath_tags_seq = page_info["xpath_tags_seq"]
        self.xpath_subs_seq = page_info["xpath_subs_seq"]
        backbone = build_dom_graph(dom)
        self.graph = nx2dgl(backbone)

    @property
    def features(self):
        with torch.no_grad():
            embeddings = encoder(self.text_nodes_texts).cpu()
            torch.cuda.empty_cache()
        return {
            "text_nodes_ids": self.text_nodes_ids,
            "text_nodes_texts": embeddings,
            "xpath_tags_seq": self.xpath_tags_seq,
            "xpath_subs_seq": self.xpath_subs_seq,
            "graph": self.graph,
        }


class SampleDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.features = [sample.features for sample in samples]
        self.labels = [sample.text_nodes_labels for sample in samples]
        assert len(self.features) == len(self.labels)
        self.len = len(self.features)

    def __getitem__(self, index) -> T_co:
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return self.len


def flatten(nested_list):
    return [item for item in chain(*nested_list)]


def collate_fn(samples: list) -> T_co:
    features, labels = zip(*samples)
    batched_labels = torch.tensor(flatten(labels), dtype=torch.long)
    ids = flatten([feature["text_nodes_ids"] for feature in features])
    texts = flatten([feature["text_nodes_texts"] for feature in features])
    xpath_tags_seq = torch.stack([feature["xpath_tags_seq"] for feature in features], dim=0)
    xpath_subs_seq = torch.stack([feature["xpath_subs_seq"] for feature in features], dim=0)
    graphs = dgl.batch([feature["graph"] for feature in features])
    batched_features = {
        "ids": ids,
        "text_embeddings": torch.stack(texts, dim=0),
        "xpath_tags_seq": xpath_tags_seq,
        "xpath_subs_seq": xpath_subs_seq,
        "graphs": graphs,
    }
    return batched_features, batched_labels



