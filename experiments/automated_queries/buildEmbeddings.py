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
#


from __future__ import print_function

import argparse
import sys
import json

# %%
import os

import numpy as np

from data_extraction.data_reader.data_loader import Loader
from data_extraction.data_reader.data_reader import Reader
from program_helper.program_reverse_map import ProgramRevMapper
from trainer_vae.infer import BayesianPredictor



class embedding_server():
    def __init__(self, save_path):

        self.encoder = BayesianPredictor(save_path, batch_size=5, seed=0)
        self.config_path = os.path.join(save_path, 'config.json')
        self.data_path = save_path

        self.prog_mapper = ProgramRevMapper(self.encoder.config.vocab)
        self.psi_list = []

        return

    def reset(self):
        self.prog_mapper.reset()
        self.psi_list.clear()

    def getEmbeddings(self, logdir):
        dump_data_path = self.data_path
        reader = Reader(dump_data_path=dump_data_path,
                        infer=True,
                        infer_vocab_path=self.config_path)
        reader.read_file(filename=logdir + '/L4TestProgramList.json')
        reader.wrangle()
        reader.log_info()
        reader.dump()

        loader = Loader(dump_data_path, self.encoder.config)
        while True:
            try:
                batch = loader.next_batch()
            except StopIteration:
                break
            psi = self.encoder.get_initial_state_from_next_batch(batch)
            psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
            self.psi_list.extend(psi_)

            self.prog_mapper.add_batched_data(batch)

        print('\nWriting to {}...'.format(''), end='\n')
        with open(logdir + '/EmbeddedProgramList.json', 'w') as f:
            json.dump({'embeddings': [psi.tolist() for psi in self.psi_list]}, fp=f, indent=2)

        return


if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel',
                        help='checkpoint model during training here')

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)
