from absl import app
from absl import flags
from tqdm import tqdm

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from ietools.utils import swde as constants
from ietools.utils.functions import load_from_pkl

FLAGS = flags.FLAGS

flags.DEFINE_string("input_swde_path", "",
                    "The root path to swde html page files.")
flags.DEFINE_string("ground_truth_path", "",
                    "The root path to swde ground truth files.")
flags.DEFINE_string("pages_info_path", "",
                    "The output path for full html pages info.")


def construct_ground_truth_mapping(swde_data, ground_truth_path):
    ground_truth_mapping = {page['id']: dict() for page in swde_data}
    vertical_to_websites_map = constants.VERTICAL_WEBSITES
    for v in vertical_to_websites_map:
        for filename in os.listdir(os.path.join(ground_truth_path, v)):
            if filename == '.DS_Store':
                continue
            vertical, website, attribute = filename.replace('.txt', '').split('-')
            with open(os.path.join(ground_truth_path, v, filename), 'r') as reader:
                lines = reader.readlines()
            ground_truths = [line.strip().split("\t") for line in lines[2:]]
            for item in ground_truths:
                idx = item[0]
                values = item[2:]
                if values[0] == '<NULL>':
                    continue
                page_id = f'{vertical}-{website}-{idx}'
                ground_truth_mapping[page_id][attribute] = values
                print(page_id, values)
    return ground_truth_mapping


def concat_and_write_pages_info(swde_data, ground_truth_path, pages_info_path):
    pages_info = swde_data
    ground_truth_mapping = construct_ground_truth_mapping(swde_data, ground_truth_path)
    for idx, page_dict in enumerate(swde_data):
        page_id = page_dict['id']
        page_ground_truth = ground_truth_mapping[page_id]
        pages_info[idx]['ground_truth'] = page_ground_truth
    with open(pages_info_path, 'wb') as fp:
        pickle.dump(pages_info, fp)


def main(_):
    swde_data = load_from_pkl(FLAGS.input_swde_path)
    concat_and_write_pages_info(swde_data, FLAGS.ground_truth_path, FLAGS.pages_info_path)


if __name__ == '__main__':
    app.run(main)
