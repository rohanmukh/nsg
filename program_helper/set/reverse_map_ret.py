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

class RetReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.ret_type = []

        self.num_data = 0
        return

    def add_data(self, ret_type):
        self.ret_type.extend(ret_type)
        self.num_data += len(ret_type)

    def get_element(self, id):
        return self.ret_type[id]

    def decode_ret(self, ret_type):
        print('--Ret Type--')
        print(self.vocab.chars_type[ret_type])

    def reset(self):
        self.ret_type = []
        self.num_data = 0