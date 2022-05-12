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
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from program_helper.infer_model_helper import InferModelHelper
from utilities.basics import conditional_director_creator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

NUM_DATA = 100
SKIP_DATA = 0
BEAM_WIDTH = 10
SEED = 500


def test_next_token_probability(_clargs):
    conditional_director_creator(clargs.temp_result_path)
    dump_data_path = os.path.join(_clargs.temp_result_path, 'test_data')
    conditional_director_creator(dump_data_path)

    infer_model = InferModelHelper(model_path=_clargs.continue_from,
                                   seed=SEED,
                                   beam_width=BEAM_WIDTH,
                                   max_num_data=NUM_DATA,
                                   depth=None,
                                   visibility=_clargs.visibility
                                   )

    infer_model.read_and_dump_data(filepath=_clargs.filepath,
                                   data_path=dump_data_path,
                                   )

    ignore_concepts = False
    horizon_prob_dict = infer_model.get_horizon_prob(
        data_path=dump_data_path,
        ignore_concepts=ignore_concepts
        )

    if ignore_concepts:
        print("Concept internal nodes are ignored")
    else:
        print("All AST nodes are considered")
    cum_log_sum = [0., 0.]
    pretty_stat = []
    for j, length in enumerate(sorted(horizon_prob_dict.keys())):
        prob = horizon_prob_dict[length]
        if any(np.isnan(prob)) or any(np.isinf(prob)):
            continue
        log_prob = np.log([p+0.00001 for p in prob])
        cum_log_sum = [p + c for p, c in zip(log_prob, cum_log_sum)]
        # norm_cum_log_sum = cum_log_sum #/ (j+1)
        pretty_stat.append((length, prob[0]))
        print("The horizon probability for {} is {} and log prob is {}".format(length, prob, log_prob))
        print(cum_log_sum)

    [print("({},{:.2f}) ".format(item[0], item[1]), end=',') for item in pretty_stat[::-1]]

    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Log Probability')
    ax1.set_title('Token position')
    line, = ax1.plot([item[0] for item in pretty_stat[::-1]] , [item[1] for item in pretty_stat[::-1]], 'b',
         linewidth=3)
    midfix = "_all_node" if not ignore_concepts else "_all_but_concept"
    plt.savefig("output" +  midfix + ".png")
    infer_model.close()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='save',
                        help='ignore config options and continue training model checkpointed here')
    # parser.add_argument('--data', default='../data_extraction/data_reader/data')
    parser.add_argument('--filepath', default=None)
    parser.add_argument('--temp_result_path', type=str, default='results/test_next_tokem/')
    parser.add_argument('--visibility', type=float, default=1.00)


    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    test_next_token_probability(clargs)
