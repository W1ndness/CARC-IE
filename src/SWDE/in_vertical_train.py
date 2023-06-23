from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from absl import app
from absl import flags
from absl import logging
import os
import sys

from transformers import MarkupLMConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from ietools.utils.functions import load_from_pkl
from ietools.utils import swde as constants
from ietools.models.dataset import Sample, SampleDataset, collate_fn
from ietools.models.ie import Model, ModelConfig
from ietools.models.train import train

FLAGS = flags.FLAGS

# args for dataset
flags.DEFINE_string("swde_path", "",
                    "The root path to swde html page files.")
flags.DEFINE_string("pack_path", "",
                    "The file path to save the packed data, done in pack_data.py")
flags.DEFINE_string("nodes_info_path", "",
                    "The path for html pages info with nodes info, done in prepare_data.py")
flags.DEFINE_string("verticals", constants.VERTICAL_WEBSITES.keys(),
                    "Verticals used in training, splits with \',\'")
flags.DEFINE_integer("num_seeds", "1",
                     "Seed websites used for training.")
flags.DEFINE_boolean("shuffle", "True", "If shuffle the dataset.")
flags.DEFINE_integer("num_workers", "4",
                     "Number of workers for training.")
flags.DEFINE_boolean("drop_last", "True", "If drop the last batch.")

# args for model
flags.DEFINE_integer("batch_size", "32",
                     "Batch size for training.")
flags.DEFINE_integer("num_epochs", "200",
                     "Number of epochs for training.")
flags.DEFINE_enum("optimizer", "Adam", ["Adam", "AdamW", "SGD"])
flags.DEFINE_float("lr", "0.001",
                   "Learning rate for training.")


def find_vertical(packed_data, vertical):
    st, ed = 0, len(packed_data)
    while st < len(packed_data) and packed_data[st]["vertical"] != vertical:
        st += 1
    while ed >= 0 and packed_data[ed - 1]["vertical"] != vertical:
        ed -= 1
    return st, ed


def find_website_pages_in_pack_data(packed_vertical_data, website, num):
    st = 0
    while st < len(packed_vertical_data) and packed_vertical_data["website"] != website:
        st += 1
    return packed_vertical_data[st:st + num]


def read_website_counts_from_list(website_list_path, vertical):
    with open(website_list_path, "r") as f:
        lines = f.readlines()
    contents = None
    for i, line in enumerate(lines):
        if line == vertical:
            contents = lines[i + 2:i + 12]
            break
    if contents is None:
        raise Exception(f"{vertical} not found in {website_list_path}")
    return {line.split(" ")[0]: int(line.split(" ")[1]) for line in contents}


def make_dataset(vertical, websites, website_counts, packed_vertical_data, split=False):
    samples = []
    for website in websites:
        count = website_counts[website]
        websites_pack_data = find_website_pages_in_pack_data(packed_vertical_data,
                                                             website,
                                                             count)
        websites_nodes_info_path = os.path.join(FLAGS.nodes_info_path,
                                                vertical,
                                                f'{vertical}-{website}.pkl')
        websites_nodes_info = load_from_pkl(websites_nodes_info_path)
        website_samples = []
        for page in websites_pack_data:
            page_id = page['id']
            page_no = page_id.split('-')[-1]
            page_html = page['html_str']
            page_info = websites_nodes_info[page_no]
            assert page_info['page_id'] == page_id
            website_samples.append(Sample(page_info, page_html, website))
        samples += website_samples
    if not split:
        return SampleDataset(samples)
    else:
        train_samples, val_samples = train_test_split(samples,
                                                      test_size=0.2,
                                                      random_state=42)
        return SampleDataset(train_samples), SampleDataset(val_samples)


def prepare_in_vertical_dataset(packed_data, vertical, num_seeds):
    if vertical not in constants.VERTICAL_WEBSITES:
        raise ValueError(f"{vertical} is not in {constants.VERTICAL_WEBSITES}")
    logging.info(f"Preparing in-vertical dataset for {vertical}")
    st, ed = find_vertical(packed_data, vertical)
    packed_vertical_data = packed_data[st:ed]
    websites = constants.VERTICAL_WEBSITES[vertical]
    seed_websites = websites[:num_seeds]  # hope the choice of site seeds does not affect the results
    test_websites = websites[num_seeds:]
    website_list_path = os.path.join(FLAGS.swde_path, "website_list.txt")
    website_counts = read_website_counts_from_list(website_list_path, vertical)
    train_dataset, val_dataset = make_dataset(vertical, seed_websites, website_counts,
                                              packed_vertical_data, split=True)
    test_dataset = make_dataset(vertical, test_websites, website_counts, packed_vertical_data)
    return train_dataset, val_dataset, test_dataset


def in_vertical_train(train_dataset, val_dataset, test_dataset, num_classes):
    train_iter = DataLoader(train_dataset,
                            batch_size=FLAGS.batch_size,
                            shuffle=FLAGS.shuffle,
                            num_workers=FLAGS.num_workers,
                            drop_last=FLAGS.drop_last,
                            collate_fn=collate_fn)
    val_iter = DataLoader(val_dataset,
                          batch_size=FLAGS.batch_size,
                          shuffle=FLAGS.shuffle,
                          num_workers=FLAGS.num_workers,
                          drop_last=FLAGS.drop_last,
                          collate_fn=collate_fn)
    test_iter = DataLoader(test_dataset,
                           batch_size=FLAGS.batch_size,
                           shuffle=FLAGS.shuffle,
                           num_workers=FLAGS.num_workers,
                           drop_last=FLAGS.drop_last,
                           collate_fn=collate_fn)
    xpath_embed_config = MarkupLMConfig()
    config = ModelConfig(text_embed_size=768,
                         gnn_out_size=768,
                         xpath_embeddings_config=xpath_embed_config,
                         gnn_activation='ReLU',
                         gnn_aggregator=None,
                         gnn_dropout=.5,
                         mlp_hidden_dims=[768],
                         mlp_activation='ReLU',
                         mlp_dropout=.5,
                         mlp_batch_norm=True,
                         num_classes=num_classes)
    model = Model(config)
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=FLAGS.lr)
    train(model, train_iter, val_iter, loss, optimizer, FLAGS.num_epochs)


def main(_):
    packed_data = load_from_pkl(FLAGS.pack_path)
    train_dataset, val_dataset, test_dataset = prepare_in_vertical_dataset(packed_data,
                                                                           FLAGS.verticals,
                                                                           FLAGS.num_seeds)


if __name__ == '__main__':
    app.run(main)
