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

import argparse
import os

from experiments.jaccard_metric.utils import plotter
from synthesis.ops.candidate_ast import API_NODE
from trainer_vae.infer import BayesianPredictor
from experiments.jaccard_metric.get_jaccard_metrics import helper

from data_extraction.data_reader.data_loader import Loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(clargs):
    num_centroids = 10

    predictor = BayesianPredictor(clargs.continue_from, batch_size=5)
    loader = Loader(clargs.data, predictor.config)
    psis, labels = [], []
    for i in range(10000):
        nodes, edges, targets, var_decl_ids, \
        node_type_numbers, \
        type_helper_val, expr_type_val, ret_type_val, \
        ret_type, fp_in, fields, \
        apicalls, types, keywords, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method  = loader.next_batch()
        psi = predictor.get_latent_state(apicalls, types, keywords,
                                            ret_type, fp_in, fields, method, classname, javadoc_kws,
                                            surr_ret, surr_fp, surr_method
                                           )
        psis.extend(psi)

        apiOrNot = node_type_numbers == API_NODE
        for t, api_bool in zip(targets, apiOrNot):
            label = get_apis(t, api_bool, predictor.config.vocab.chars_api)
            labels.append(label)

    predictor.close()

    new_states, new_labels = [], []
    for state, label in zip(psis, labels):
        if len(label) != 0:
            new_states.append(state)
            new_labels.append(label)

    print('API Call Jaccard Calculations')
    jac_api_matrix, jac_api_vector = helper(new_states, new_labels, num_centroids=num_centroids)
    plotter(jac_api_matrix, jac_api_vector, name=clargs.filename)

    return


def get_apis(calls, apiOrNot, vocab):
    apis = []
    for call, api_bool in zip(calls, apiOrNot):
        if api_bool and call > 0:
            api = vocab[call]
            apis.append(api)

    return apis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--continue_from', type=str, default='save',
                        help='directory to load model from')
    parser.add_argument('--top', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--data', default='../data_extraction/data_reader/data')
    clargs = parser.parse_args()
    clargs.folder = 'results/test_jaccard/'
    if not os.path.exists(clargs.folder):
        os.makedirs(clargs.folder)
    clargs.filename = clargs.folder + 'jaccard_' + clargs.continue_from
    main(clargs)
