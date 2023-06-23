# one page in pickle dumped from <pack_data.py>
# <class 'dict'>
# keys: id, vertical, website, path, html_str

# parse one page to another pickle file, contains the information of one page from DOM tree
# use pre-order to traverse DOM tree
# pickle contains a list of the information of DOM nodes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import re
import warnings

from absl import app
from absl import flags
from absl import logging
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from ietools.utils import swde as constants
from ietools.utils.functions import get_dom_tree, load_from_pkl

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings('ignore')
FLAGS = flags.FLAGS

flags.DEFINE_string("swde_path", "",
                    "The root path to swde html page files.")
flags.DEFINE_string("pages_info_path", "",
                    "The input path for full html pages info.")
flags.DEFINE_string("nodes_info_path", "",
                    "The output path for html pages info with nodes info.")
flags.DEFINE_string("model_name", "",
                    "Model name of the text encoder.")


def element_xpath(element):
    """
    Get the XPath of an element using lxml.

    :param element: The lxml element for which to get the XPath.
    :return: A tuple containing two lists, the first being a list of tag names in order from root to target,
             and second being a list of indices indicating the position of the target element among its siblings.
    """
    xpath_tags = []
    xpath_subscripts = []
    while element is not None:
        xpath_tags.append(element.tag)
        siblings = element.xpath('following-sibling::*[self::p]') + [element] + element.xpath(
            'preceding-sibling::*[self::p]')
        if len(siblings) == 1:
            xpath_subscripts.insert(0, 0)
        else:
            index = siblings.index(element)
            xpath_subscripts.insert(0, index + 1)
        element = element.getparent()
    return xpath_tags, xpath_subscripts


def construct_xpath(xpath_tags, xpath_subscripts):
    xpath = ""
    for tagname, subs in zip(xpath_tags, xpath_subscripts):
        xpath += f"/{tagname}"
        if subs != 0:
            xpath += f"[{subs}]"
    return xpath


def match_node_label(content, mapping):
    if content in mapping:
        return mapping[content]
    return 'None'


def parse_dom_tree(dom, page_info):
    dense_nodes = []
    ground_truth = page_info['ground_truth']
    inverse_mapping = dict()
    for key, values in ground_truth.items():
        mapping = {value: key for value in values}
        inverse_mapping.update(mapping)
    xpath_tags_seq, xpath_subs_seq = [], []
    cnt, tot, num_nodes = 0, 0, 0
    for idx, element in enumerate(dom.iter()):
        num_nodes += 1
        xpath_tags, xpath_subs = element_xpath(element)
        xpath_tags_seq.append(xpath_tags)
        xpath_subs_seq.append(xpath_subs)
        if element.text is None:
            continue
        content = element.text.strip()
        if content != "":
            tot += 1
            if len(element.getchildren()) == 0:  # @test: if all nodes with texts are leaf nodes
                # print(idx, "Leaf Node", content)
                cnt += 1
            label = match_node_label(content, inverse_mapping)
            if label != 'None':
                info = dict(idx=idx, label=label)
                dense_nodes.append(info)
    print(f"Leaf nodes/All nodes: {cnt}/{tot}")
    print(f"Number of nodes: {num_nodes}")

    return dense_nodes, xpath_tags_seq, xpath_subs_seq


def generate_nodes_info_and_write_to_file(swde_path, pages_info_path, nodes_info_path):
    if not os.path.exists(nodes_info_path):
        os.makedirs(nodes_info_path)
    pages_info = load_from_pkl(pages_info_path)
    print(f"Load pages info from {pages_info_path}, #pages: {len(pages_info)}")

    # page info: id, vertical, website, html_str, ground_truth
    vertical_to_websites_map = constants.VERTICAL_WEBSITES
    total = 0
    for v in vertical_to_websites_map:
        if not os.path.exists(os.path.join(nodes_info_path, v)):
            os.makedirs(os.path.join(nodes_info_path, v))
        for w in os.listdir(os.path.join(swde_path, v)):
            if w == '.DS_Store':
                continue
            website = re.search('(.*)-(.*)\((.*)\)', w).groups()[1]
            filenames = os.listdir(os.path.join(swde_path, v, w))
            filenames = [f for f in filenames if f.endswith('.htm') or f.endswith('.html')]
            st = total
            ed = st + len(filenames)
            total += len(filenames)
            print(f"Parsing {v}-{website}, from {st} to {ed}, {len(filenames)} pages.")
            print(f"Saving path: {os.path.join(nodes_info_path, v, f'{v}-{website}.pkl')}")
            website_nodes_info = []
            with tqdm(total=len(filenames), file=sys.stdout, desc=f"parse {v}-{website}") as progressbar:
                for page_info in pages_info[st:ed]:
                    page_id = page_info['id']
                    assert v in page_id and website in page_id
                    dom = get_dom_tree(page_info['html_str'], page_info['website'])
                    nodes_info, xpath_tags_seq, xpath_subs_seq = parse_dom_tree(dom, page_info)
                    website_nodes_info.append(dict(page_id=page_id,
                                                   nodes_info=nodes_info,
                                                   xpath_tags_seq=xpath_tags_seq,
                                                   xpath_subs_seq=xpath_subs_seq))
                    progressbar.update()
            # with tqdm(total=len(filenames), file=sys.stdout, desc=f"embed {v}-{website}") as progressbar, \
            #         torch.no_grad():
            #     torch.cuda.empty_cache()
            #     for page in website_nodes_info:
            #         page_id = page['page_id']
            #         assert v in page_id and website in page_id
            #         batched_texts = [node['content'] for node in page['nodes_info']]
            #         batched_embeddings = encoder(batched_texts).cpu()
            #         assert batched_embeddings.size()[0] == len(page['nodes_info'])
            #         batch_no = 0
            #         for node in page['nodes_info']:
            #             node['embedding'] = batched_embeddings[batch_no]
            #             batch_no += 1
            #         progressbar.update()
            with open(os.path.join(nodes_info_path, v, f'{v}-{website}.pkl'), 'wb') as fp:
                pickle.dump(website_nodes_info, fp)
            print(f"Successfully write {v}-{website} to {os.path.join(nodes_info_path, v, f'{v}-{website}.pkl')}.")


def main(_):
    generate_nodes_info_and_write_to_file(FLAGS.swde_path,
                                          FLAGS.pages_info_path,
                                          FLAGS.nodes_info_path)


if __name__ == '__main__':
    app.run(main)
