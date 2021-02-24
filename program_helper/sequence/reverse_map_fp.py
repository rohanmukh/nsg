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

class FPReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.fp_in = []

        self.num_data = 0
        return

    def add_data(self, fp_in):

        self.fp_in.extend(fp_in)
        self.num_data += len(fp_in)

    def get_element(self, id):
        return self.fp_in[id]

    def decode_fp_paths(self, fp_element):
        print('--Formal Params--')
        fp_in = fp_element
        for fp_type in fp_in:
            print(self.vocab.chars_type[fp_type], end=',')
        print()

    def reset(self):
        self.fp_in = []
        self.num_data = 0