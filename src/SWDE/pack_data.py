# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DONE READ!
#



r"""To pack all the swde html page files into a single pickle file.

This script is to generate a single file to pack up all the content of htmls.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from absl import app
from absl import flags
import tqdm
import ietools.utils.swde as constants

FLAGS = flags.FLAGS

# Flags related to input data.
flags.DEFINE_string("input_swde_path", "",
                    "The root path to swde html page files.")
flags.DEFINE_string("output_pack_path", "",
                    "The file path to save the packed data.")
flags.DEFINE_integer("first_n_pages", -1,
                     "The cut-off number to shorten the number of pages.")


def pack_swde_data(swde_path, pack_path, cut_off):
    """Packs the swde dataset to a single file.

    Args:
      swde_path: The path to SWDE dataset pages (http://shortn/_g22KuARPAi).
      pack_path: The path to save packed SWDE dataset file.
      cut_off: To shorten the list for testing.
    Returns:
      None
    """
    # Get all website names for each vertical.
    #   The SWDE dataset fold is structured as follows:
    #     - swde/                                    # The root folder.
    #       - swde/auto/                             # A certain vertical.
    #         - swde/auto/auto-aol(2000)/            # A certain website.
    #           - swde/auto/auto-aol(2000)/0000.htm  # A page.
    # Get all vertical names.
    vertical_to_websites_map = constants.VERTICAL_WEBSITES
    """
    for `auto`, that is --->
    [
        "msn", "aol", "kbb", "cars", "yahoo", "autoweb", "autobytel",
        "automotive", "carquotes", "motortrend"
    ]
    """
    # The data dict initialized with the path of each html file of SWDE.
    swde_data = list()
    print("Start loading data...")
    for v in vertical_to_websites_map:
        for w in os.listdir(os.path.join(swde_path, v)):
            if w == '.DS_Store':
                continue
            page_count = 0
            filenames = os.listdir(os.path.join(swde_path, v, w))
            filenames.sort()
            for filename in filenames:
                if not filename.endswith('.htm'):
                    continue
                print(os.path.join(swde_path, v, w, filename))
                website = re.search('(.*)-(.*)\((.*)\)', w).groups()[1]
                page_id = f'{v}-{website}-{filename[:4]}'
                print(page_id)
                page = dict(id=page_id, vertical=v, website=website, path=os.path.join(v, w, filename))
                # path is something like `book/book-amazon(2000)/0000.htm`
                swde_data.append(page)
                page_count += 1
                if 0 < cut_off == page_count:
                    break

    # Load the html data.
    with tqdm.tqdm(total=len(swde_data), file=sys.stdout) as progressbar:
        for page in swde_data:
            with open(os.path.join(swde_path, page["path"]), encoding='utf-8') as webpage:
                page["html_str"] = webpage.read()

            progressbar.set_description("processed")
            progressbar.update(1)

    # now, the swde_data is a list
    # for each page in it
    # we have it as
    # {"vertical":'book',"website":'book-amazon(2000)',"path:'book/book-amazon(2000)/0000.htm',"html_str":xx}


    # and finally these info are dumped into the swde.pickle file

    # Save the html_str data.
    with open(pack_path, "wb") as gfo:
        pickle.dump(swde_data, gfo)


def main(_):
    pack_swde_data(
        swde_path=FLAGS.input_swde_path,
        pack_path=FLAGS.output_pack_path,
        cut_off=FLAGS.first_n_pages)


if __name__ == "__main__":
    app.run(main)
