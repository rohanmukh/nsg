# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import json
import sys
import textwrap
import os

from data_extraction.data_reader.utils import read_vocab
from program_helper.program_reverse_map import ProgramRevMapper
from data_extraction.data_reader.data_loader import Loader
from data_extraction.data_reader.data_reader import MAX_AST_DEPTH, MAX_FP_DEPTH, MAX_KEYWORDS

HELP = """"""

NUM_DATA = 100
BATCH_SIZE = 10

def data_checker(_clargs):
    config_file = os.path.join(_clargs.data, 'vocab.json')
    with open(config_file) as f:
        config = read_vocab(json.load(f))
    config.max_ast_depth = MAX_AST_DEPTH
    config.input_fp_depth = MAX_FP_DEPTH
    config.max_keywords = MAX_KEYWORDS
    config.batch_size = BATCH_SIZE
    config.trunct_num_batch = None

    loader = Loader(clargs.data, config)
    prog_mapper = ProgramRevMapper(config.vocab)
    for i in range(NUM_DATA // config.batch_size):
        nodes, edges, targets, var_decl_ids, return_reached,\
        node_type_number, \
        type_helper_val, expr_type_val, ret_type_val, \
        all_var_mappers, iattrib, \
        ret_type, fp_in, fields, \
        apicalls, types, keywords, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method = loader.next_batch()
        prog_mapper.add_data(nodes, edges, targets, var_decl_ids, \
                             node_type_number, return_reached,\
                             type_helper_val, expr_type_val, ret_type_val, \
                             all_var_mappers,
                             ret_type, fp_in, \
                             fields, \
                             apicalls, types, keywords, method, classname, javadoc_kws, \
                             surr_ret, surr_fp, surr_method)

    for i in range(NUM_DATA):
        print("Starting to decode program with id {}".format(i))
        prog_mapper.decode_paths(i, partial=False)
        print()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data', default='./data')

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    data_checker(clargs)
