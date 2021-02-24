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

class SurroundingReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.ret_type = []
        self.fp_nodes = []
        self.method = []

        self.num_data = 0
        return

    def add_data(self, surr_ret, surr_fp, surr_method):
        self.ret_type.extend(surr_ret)
        self.fp_nodes.extend(surr_fp)
        self.method.extend(surr_method)

        self.num_data += len(surr_ret)

    def get_element(self, id):
        return (self.ret_type[id], self.fp_nodes[id], self.method[id])

    def decode(self, elem_tuple):
        ret_type, fp_nodes, methods = elem_tuple
        print('--Ret Type--')
        for t in ret_type:
            print(self.vocab.chars_type[t])
        print('--FP Types--')
        for fp_node in fp_nodes:
            for node in fp_node:
                print(self.vocab.chars_type[node], end=',')
            print()
        print('--Method--')
        for method in methods:
            for kw in method:
                print(self.vocab.chars_kw[kw], end=',')
            print()

    def reset(self):
        self.ret_type = []
        self.fp_nodes = []
        self.method = []

        self.num_data = 0