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

class FieldReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.field_ins = []

        self.num_data = 0
        return

    def add_data(self, field_in):
        self.field_ins.extend(field_in)
        self.num_data += len(field_in)

    def get_element(self, id):
        return self.field_ins[id]

    def decode_fp_paths(self, field_element):
        print('--Field Params--')
        field_in = field_element

        for f_type in field_in:
            print(self.vocab.chars_type[f_type], end=',')
        print()

    def reset(self):
        self.field_ins = []

        self.num_data = 0