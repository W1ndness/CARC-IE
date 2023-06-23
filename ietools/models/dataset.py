from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import dgl

from ..utils.functions import get_dom_tree
from .graph import build_dom_graph, nx2dgl
from ..models.encoder import TextEncoder


def get_texts(dom, text_nodes_ids):
    return [element.text.strip() for idx, element in enumerate(dom) if idx in text_nodes_ids]


class Sample:
    def __init__(self, page_info: dict, page_html, website):
        nodes_info = page_info["nodes_info"]
        self.text_nodes_ids = [node["id"] for node in nodes_info]
        dom = get_dom_tree(page_html, website)
        self.text_nodes_texts = get_texts(dom, self.text_nodes_ids)
        self.text_nodes_labels = [node["label"] for node in nodes_info]
        self.all_xpath_tags_seq = [node["xpath_tag_seq"] for node in nodes_info]
        self.all_xpath_subs_seq = [node["xpath_sub_seq"] for node in nodes_info]
        backbone = build_dom_graph(dom)
        self.graph = nx2dgl(backbone)

    @property
    def features(self):
        encoder = TextEncoder()
        with torch.no_grad():
            embeddings = encoder(self.text_nodes_texts).cpu()
        return {
            "text_nodes_ids": self.text_nodes_ids,
            "text_nodes_texts": embeddings,
            "all_xpath_tags_seq": self.all_xpath_tags_seq,
            "all_xpath_subs_seq": self.all_xpath_subs_seq,
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


def collate_fn(samples: list) -> T_co:
    features, labels = zip(*samples)
    if not isinstance(labels, torch.Tensor):
        batched_labels = torch.Tensor(labels)
    else:
        batched_labels = labels
    ids = [features["text_nodes_ids"] for feature in features]
    texts = [feature["text_nodes_texts"] for feature in features]
    xpath_tags_seq = [feature["all_xpath_tags_seq"] for feature in features]
    xpath_subs_seq = [feature["all_xpath_subs_seq"] for feature in features]
    graphs = dgl.batch([feature["graph"] for feature in features])
    batched_features = {
        "ids": ids,
        "text_embeddings": torch.stack(texts, dim=0),
        "xpath_tags_seq": xpath_tags_seq,
        "xpath_subs_seq": xpath_subs_seq,
        "graphs": graphs,
    }
    return batched_features, batched_labels



