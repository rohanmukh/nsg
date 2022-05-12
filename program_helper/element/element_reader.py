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

import numpy as np
import pickle

from program_helper.ast.ops import DType
from utilities.vocab_building_dictionary import VocabBuildingDictionary


class ElementReader:

    def __init__(self, vocab=None, infer=True):
        self.vocab = vocab
        self.infer = infer
        self.return_types_np = None
        self.ret_head = None
        return

    def read_while_vocabing(self, program_ret_js):
        self.ret_head = DType(program_ret_js)
        return_type_id = self.vocab.conditional_add_or_get_node_val(program_ret_js, self.infer)
        return return_type_id

    # sz is total number of data points, Wrangle the Return Types
    def wrangle(self, return_types, min_num_data=None):
        if min_num_data is None:
            sz = len(return_types)
        else:
            sz = max(min_num_data, len(return_types))

        self.return_types_np = np.zeros(sz, dtype=np.int32)
        for i, rt in enumerate(return_types):
            self.return_types_np[i] = rt
        return

    def save(self, path):
        with open(path + '/return_types.pickle', 'wb') as f:
            pickle.dump(self.return_types_np, f)

    def load_data(self, path):
        with open(path + '/return_types.pickle', 'rb') as f:
            self.return_types_np = pickle.load(f)
        return

    def truncate(self, sz):
        self.return_types_np = self.return_types_np[:sz]

    def split(self, num_batches):
        self.return_types_np = np.split(self.return_types_np, num_batches, axis=0)

    def get(self):
        return self.return_types_np


    def add_data_from_another_reader(self, element_reader):
        if self.return_types_np is None:
            self.return_types_np = element_reader.return_types_np
        else:
            self.return_types_np = np.append(self.return_types_np, element_reader.return_types_np)

