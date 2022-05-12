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

import os
import argparse
import sys

from data_extraction.data_reader.data_loader import Loader
from experiments.tSNE_visualizor.get_labels import get_api
from experiments.tSNE_visualizor.tSNE import fitTSNEandplot
from synthesis.ops.candidate_ast import API_NODE
from trainer_vae.infer import BayesianPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def plot(clargs):
    predictor = BayesianPredictor(clargs.continue_from, batch_size=5)
    loader = Loader(clargs.data, predictor.config)
    states, labels = [], []
    for i in range(min(loader.config.num_batches, 20000)):
        nodes, edges, targets, var_decl_ids, \
        node_type_number, \
        type_helper_val, expr_type_val, ret_type_val, \
        ret_type, fp_in, fields, \
        apicalls, types, keywords, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method = loader.next_batch()
        state = predictor.get_latent_state(apicalls, types, keywords,
                                            ret_type, fp_in, fields, method, classname, javadoc_kws,
                                            surr_ret, surr_fp, surr_method
                                           )
        states.extend(state)

        apiOrNot = node_type_number == API_NODE
        for t, api_bool in zip(targets, apiOrNot):
            label = get_api(predictor.config, t, api_bool)
            labels.append(label)
    predictor.close()

    new_states, new_labels = [], []
    for state, label in zip(states, labels):
        if label != 'N/A':
            new_states.append(state)
            new_labels.append(label)
    print('Number of effective program:: ' + str(len(new_labels)))
    print('Fitting tSNE')
    fitTSNEandplot(new_states, new_labels, clargs.filename)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='save',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--topK', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--data', type=str, default='../data_extraction/data_reader/data',
                        help='load data from here')
    clargs = parser.parse_args()
    clargs.folder = 'results/test_visualize/'
    clargs.filename = clargs.folder + 'plot_' + clargs.continue_from + '.png'
    if not os.path.exists(clargs.folder):
        os.makedirs(clargs.folder)
    sys.setrecursionlimit(clargs.python_recursion_limit)

    plot(clargs)
